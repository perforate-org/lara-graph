use super::class::{CLASS_BYTES, CLASS_PER_PAGE, CODES};
use super::*;
use crate::store::memory::VecMemory;
use alloc::collections::BTreeSet;

#[derive(Default)]
struct Backing {
    pages: VecMemory,
    directory: VecMemory,
}

impl Backing {
    fn create(&self) -> PageStore<VecMemory> {
        PageStore::format(self.pages.clone(), self.directory.clone()).unwrap()
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
    for (pool, at) in [Pool::Edges, Pool::Lists]
        .into_iter()
        .flat_map(|pool| (0..ByteClass::CHUNK_CLASSES).map(move |at| (pool, at)))
    {
        let class = ByteClass(u8::try_from(at).unwrap());
        let mut prev = None;
        let mut cursor = store.partial_head(lane(pool, class));
        while let Some(page) = cursor {
            let PageRecord::Chunk {
                class: c,
                pool: p,
                used,
                prev: back,
                next,
                ..
            } = store.record(page)
            else {
                panic!("non-chunk on partial list")
            };
            assert_eq!(
                (c, p, back),
                (class, pool, prev),
                "partial list class, pool and back link"
            );
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
                let allocated = store.allocated_in(page, class, Slots::decode(class, bitmap));
                assert_eq!(allocated.len(), usize::from(used), "slots match use count");
                assert!(used > 0, "an empty chunk is freed");
                assert_eq!(
                    partial.contains(&page),
                    used < class.per_page(),
                    "partial listing"
                );
                for index in allocated {
                    used_slots.insert((page, index, class));
                }
            }
        }
    }
    let held: BTreeSet<_> = live.iter().map(|e| (e.page, e.index(), e.class)).collect();
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
fn each_pool_fills_chunk_pages_of_its_own() {
    let backing = Backing::default();
    let mut store = backing.create();
    let edges: Vec<_> = (0..3).map(|_| store.allocate(class(64)).unwrap()).collect();
    let lists: Vec<_> = (0..3)
        .map(|_| store.lists().allocate(class(64)).unwrap())
        .collect();
    let edge = store.allocate(class(64)).unwrap();
    assert!(edges.iter().chain([&edge]).all(|e| e.page == PageId(0)));
    assert!(
        lists.iter().all(|e| e.page == PageId(1)),
        "lists keep to their page"
    );
    let all: Vec<_> = edges.iter().chain(&lists).chain([&edge]).copied().collect();
    check(&store, &all);
    // A release finds its pool from the page, whichever pool is lent.
    store.lists().release(edge);
    for e in edges {
        store.release(e);
    }
    assert_eq!(
        store.allocate(class(64)).unwrap().page,
        PageId(0),
        "the edge page is free again"
    );
    for e in lists {
        store.release(e);
    }
    check(&store, &[Extent::at(PageId(0), 0, class(64))]);
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
    let mut store = PageStore::open(backing.pages.clone(), backing.directory.clone()).unwrap();
    check(&store, &live);
    for e in live {
        store.release(e);
    }
    check(&store, &[]);
    let foreign = VecMemory::new();
    assert!(matches!(
        PageStore::open(foreign.clone(), foreign.clone()),
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

/// One random allocation or release. Half the allocations are 16-byte ones, so that slabs fill,
/// free and reuse extents; the rest are of any size.
fn random_step<M: Memory>(store: &mut PageStore<M>, live: &mut Vec<Extent>, rng: &mut Rng) {
    if live.is_empty() || rng.below(100) < 55 {
        let bytes = if rng.below(2) == 0 {
            16
        } else {
            1 + rng.below(PAGE_BYTES)
        };
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

/// Extents round-trip at the edges of every field, and the holder's spare bits do not change
/// what decodes.
#[test]
fn extents_encode_in_forty_bits() {
    let last = PageId::new(MAX_PAGES - 1);
    for extent in [
        Extent::whole(PageId::new(0)),
        Extent::whole(last),
        Extent::new(last, 127, ByteClass::SMALLEST),
        Extent::new(last, 128, ByteClass::SMALLEST),
        Extent::new(last, 255, ByteClass::SMALLEST),
        Extent::new(last, 127, ByteClass::B32),
        Extent::new(PageId::new(12_345), 3, ByteClass::for_bytes(512).unwrap()),
    ] {
        let mut bytes = extent.encode();
        assert_eq!(Extent::decode(bytes), extent);
        assert_eq!(bytes[EXTENT_BYTES - 1] & EXTENT_SPARE_BITS, 0);
        bytes[EXTENT_BYTES - 1] |= EXTENT_SPARE_BITS;
        assert_eq!(Extent::decode(bytes), extent);
    }
}

/// Each size maps to the smallest class holding it, three-quarter classes included.
#[test]
fn sizes_take_the_smallest_class_holding_them() {
    for bytes in 1..=PAGE_BYTES {
        let class = class(bytes);
        assert!(class.bytes() >= bytes, "{bytes} fits its class");
        let below = CLASS_BYTES
            .iter()
            .map(|&b| u32::from(b))
            .filter(|&b| b < class.bytes());
        assert!(
            below.into_iter().all(|b| b < bytes),
            "{bytes} takes the smallest class"
        );
    }
    assert_eq!(ByteClass::for_bytes(PAGE_BYTES + 1), None);
    for (&bytes, &count) in CLASS_BYTES.iter().zip(&CLASS_PER_PAGE) {
        assert_eq!(
            u32::from(count),
            PAGE_BYTES / u32::from(bytes),
            "{bytes}: extents per page"
        );
    }
}

/// The 16-byte and three-quarter classes encode, pack their pages and free them like the others:
/// a slab hands out all 256 16-byte extents of its page.
#[test]
fn small_and_three_quarter_classes_fill_their_pages() {
    let backing = Backing::default();
    let mut store = backing.create();
    for bytes in [16, 48, 96, 192, 384, 768] {
        let class = class(bytes);
        let extents: Vec<Extent> = (0..class.per_page())
            .map(|_| store.allocate(class).unwrap())
            .collect();
        let pages: BTreeSet<PageId> = extents.iter().map(|e| e.page()).collect();
        assert_eq!(pages.len(), 1, "{bytes}: a page's worth shares one page");
        for &e in &extents {
            assert_eq!(Extent::decode(e.encode()), e);
            assert!(
                e.offset().get() + u64::from(bytes)
                    <= e.page().offset().get() + u64::from(PAGE_BYTES)
            );
        }
        check(&store, &extents);
        for e in extents {
            store.release(e);
        }
        check(&store, &[]);
    }
}

/// The decoding table agrees with the class tables: codes 0 and 1 the 16-byte class's halves,
/// 2 to 14 each other class, 15 a page.
#[test]
fn class_codes_match_the_class_tables() {
    for (code, entry) in CODES.iter().enumerate() {
        let id = match code {
            0 | 1 => 0,
            15 => ByteClass::PAGE.id(),
            _ => u8::try_from(code - 1).unwrap(),
        };
        let class = ByteClass::from_id(id).unwrap();
        assert_eq!(entry.class, class, "code {code}");
        assert_eq!(entry.last, class.per_page() - 1, "code {code}");
        assert_eq!(u32::from(entry.bytes), class.bytes(), "code {code}");
        assert_eq!(entry.high, if code == 1 { 128 } else { 0 }, "code {code}");
    }
    assert_eq!(CLASS_BYTES.get(13), Some(&4096));
}
