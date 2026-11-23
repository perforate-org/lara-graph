use super::*;
use crate::memory::VecMemory;
use alloc::collections::BTreeSet;

#[derive(Default)]
struct Backing {
    pages: VecMemory,
    directory: VecMemory,
}

impl Backing {
    fn create(&self) -> PageStore<&VecMemory> {
        PageStore::format(&self.pages, &self.directory).unwrap()
    }
}

fn class(bytes: u32) -> ByteClass {
    ByteClass::for_bytes(bytes).unwrap()
}

/// Checks every structural invariant against the extents the test holds.
fn check<M: Memory>(store: &PageStore<M>, live: &[Extent]) {
    let minted = store.minted();
    let mut free = BTreeSet::new();
    let mut cursor = store.link(AT_FREE_HEAD);
    while let Some(page) = cursor {
        assert!(free.insert(page), "free list cycles");
        let PageRecord::Free { next } = store.record(page) else {
            panic!("live page on free list")
        };
        cursor = next;
    }
    let mut partial = BTreeSet::new();
    for lane in 0..ByteClass::CHUNK_CLASSES {
        let class = ByteClass(ByteClass::MIN_LOG2 + u8::try_from(lane).unwrap());
        let mut prev = None;
        let mut cursor = store.partial_head(class);
        while let Some(page) = cursor {
            let PageRecord::Chunk {
                class: c,
                used,
                prev: back,
                next,
                ..
            } = store.record(page)
            else {
                panic!("non-chunk on partial list")
            };
            assert_eq!((c, back), (class, prev), "partial list class and back link");
            assert!(
                used > 0 && used < class.per_page(),
                "only partially used chunks are listed"
            );
            assert!(partial.insert(page), "partial list cycles");
            prev = Some(page);
            cursor = next;
        }
    }
    let mut used_slots = BTreeSet::new();
    for page in (0..minted).map(PageId) {
        match store.record(page) {
            PageRecord::Free { .. } => assert!(free.contains(&page), "free page off the free list"),
            PageRecord::Whole => {
                used_slots.insert((page, 0, ByteClass::PAGE));
            }
            PageRecord::Chunk {
                class,
                used,
                bitmap,
                ..
            } => {
                assert_eq!(
                    bitmap.count_ones(),
                    u32::from(used),
                    "bitmap matches use count"
                );
                assert!(used > 0, "an empty chunk is freed");
                assert_eq!(
                    partial.contains(&page),
                    used < class.per_page(),
                    "partial listing"
                );
                for index in (0..class.per_page()).filter(|i| bitmap & (1 << i) != 0) {
                    used_slots.insert((page, index, class));
                }
            }
        }
    }
    let held: BTreeSet<_> = live.iter().map(|e| (e.page, e.index, e.class)).collect();
    assert_eq!(
        held, used_slots,
        "allocated extents are exactly the ones held"
    );
}

#[test]
fn chunks_share_a_page_and_free_it_when_empty() {
    let backing = Backing::default();
    let mut store = backing.create();
    let small: Vec<_> = (0..128)
        .map(|_| store.allocate(class(32)).unwrap())
        .collect();
    assert!(
        small.iter().all(|e| e.page == PageId(0)),
        "128 x 32 B fill one page"
    );
    let next = store.allocate(class(32)).unwrap();
    assert_eq!(next.page, PageId(1));
    check(&store, &[small.as_slice(), &[next]].concat());
    store.release(next);
    for e in small {
        store.release(e);
    }
    check(&store, &[]);
    // Freed chunk pages serve any class, including whole pages.
    let whole = store.allocate(ByteClass::PAGE).unwrap();
    assert!(whole.page.get() < 2, "a freed page is reused");
    assert_eq!(store.minted(), 2);
}

#[test]
fn extents_do_not_overlap_and_round_trip() {
    let backing = Backing::default();
    let mut store = backing.create();
    let a = store.allocate(class(100)).unwrap();
    let b = store.allocate(class(100)).unwrap();
    assert_eq!(a.class.bytes(), 128);
    assert_eq!(b.offset().get(), a.offset().get() + 128);
    assert_eq!(Extent::decode(b.encode()), b);
    store.write(a.offset(), &[7; 128]);
    assert_eq!(store.pages().read(a.offset(), 128), [7; 128]);
}

#[test]
fn reopen_reads_the_same_store() {
    let backing = Backing::default();
    let live = {
        let mut store = backing.create();
        let live: Vec<_> = [32, 64, 4096, 2048]
            .map(|b| store.allocate(class(b)).unwrap())
            .to_vec();
        check(&store, &live);
        live
    };
    let mut store = PageStore::open(&backing.pages, &backing.directory).unwrap();
    check(&store, &live);
    for e in live {
        store.release(e);
    }
    check(&store, &[]);
    let foreign = VecMemory::new();
    assert!(matches!(
        PageStore::open(&foreign, &foreign),
        Err(OpenError::NotAPageStore)
    ));
}

#[test]
#[should_panic(expected = "not allocated")]
fn double_release_is_corruption() {
    let backing = Backing::default();
    let mut store = backing.create();
    let a = store.allocate(class(64)).unwrap();
    let _b = store.allocate(class(64)).unwrap();
    store.release(a);
    store.release(a);
}

/// xorshift64*: deterministic, dependency-free randomness.
struct Rng(u64);

impl Rng {
    fn below(&mut self, n: u32) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        u32::try_from(self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33).unwrap() % n
    }
}

/// One random allocation (any class) or release.
fn random_step<M: Memory>(store: &mut PageStore<M>, live: &mut Vec<Extent>, rng: &mut Rng) {
    if live.is_empty() || rng.below(100) < 55 {
        let bytes = 1 << (5 + rng.below(8));
        live.push(store.allocate(class(bytes)).unwrap());
    } else {
        let at = usize::try_from(rng.below(u32::try_from(live.len()).unwrap())).unwrap();
        store.release(live.swap_remove(at));
    }
}

#[test]
fn random_operations_keep_every_invariant() {
    for seed in 1u64..=6 {
        let backing = Backing::default();
        let mut store = backing.create();
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut live: Vec<Extent> = Vec::new();
        for step in 1..=4_000 {
            random_step(&mut store, &mut live, &mut rng);
            if step % 250 == 0 {
                check(&store, &live);
            }
        }
        for e in live.drain(..) {
            store.release(e);
        }
        check(&store, &[]);
    }
}
