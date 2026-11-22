use super::*;
use crate::memory::VecMemory;
use alloc::vec::Vec;

fn len(n: u32) -> SpanLen {
    SpanLen::new(n).unwrap()
}

fn span(start: u64, n: u32) -> SlotRange {
    SlotRange::new(Slot::new(start), len(n).count()).unwrap()
}

/// What a segment looks like from outside: its range and whether it is free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seen {
    Allocated(SlotRange),
    Free(SlotRange),
}

/// Checks every structural invariant and returns the segments in address order.
fn check<M: Memory>(a: &TagAllocator<M>) -> Vec<Seen> {
    let seen = check_address_order(a);
    let free = seen.iter().filter(|s| matches!(s, Seen::Free(_))).count();
    assert_eq!(
        check_free_lists(a),
        free,
        "every free segment is listed exactly once"
    );
    let segments = u32::try_from(seen.len()).unwrap();
    assert_eq!(
        segments + check_recycled(a),
        a.record_count(),
        "every record is accounted for"
    );
    seen
}

/// The address list tiles `[0, frontier)` with fully coalesced segments.
fn check_address_order<M: Memory>(a: &TagAllocator<M>) -> Vec<Seen> {
    let mut seen = Vec::new();
    let mut previous: Option<(TagId, Segment)> = None;
    let mut cursor = a.link(AT_FIRST);
    while let Some(id) = cursor {
        let seg = a.segment(id);
        let expected_start = previous.map_or(Slot::new(0), |(_, prev)| prev.range().end());
        assert_eq!(
            seg.start, expected_start,
            "address list has a gap or overlap"
        );
        assert_eq!(seg.before, previous.map(|(id, _)| id), "broken back link");
        let mergeable = |prev: Segment| prev.is_free() && prev.len.join(seg.len).is_some();
        let uncoalesced = seg.is_free() && previous.is_some_and(|(_, prev)| mergeable(prev));
        assert!(!uncoalesced, "uncoalesced neighbours");
        seen.push(if seg.is_free() {
            Seen::Free(seg.range())
        } else {
            Seen::Allocated(seg.range())
        });
        previous = Some((id, seg));
        cursor = seg.after;
    }
    assert_eq!(a.link(AT_LAST), previous.map(|(id, _)| id), "last link");
    let end = previous.map_or(Slot::new(0), |(_, seg)| seg.range().end());
    assert_eq!(a.frontier(), end, "frontier is the end of the last segment");
    seen
}

/// Each free list holds free segments of its own class with consistent back links, and the
/// bitmaps mark exactly the non-empty lists. Returns how many segments are listed.
fn check_free_lists<M: Memory>(a: &TagAllocator<M>) -> usize {
    let mut listed = 0;
    let mut first_level = 0u32;
    for fl in 0..FL_COUNT {
        let mut second = 0u32;
        for sl in 0..class::SL_COUNT {
            let members = check_class_list(a, fl * class::SL_COUNT + sl);
            if members > 0 {
                second |= 1 << sl;
            }
            listed += members;
        }
        assert_eq!(a.second_level(fl), second, "second-level bitmap {fl}");
        if second != 0 {
            first_level |= 1 << fl;
        }
    }
    assert_eq!(
        a.read_u32(AT_FIRST_LEVEL),
        first_level,
        "first-level bitmap"
    );
    listed
}

fn check_class_list<M: Memory>(a: &TagAllocator<M>, index: u32) -> usize {
    let mut members = 0;
    let mut prev = None;
    let mut cursor = decode_link(a.region().read_array(head_offset_at(index)));
    while let Some(id) = cursor {
        let seg = a.segment(id);
        let SegmentUse::Free { prev: back, next } = seg.use_ else {
            panic!("allocated segment on a free list")
        };
        assert_eq!(back, prev, "free-list back link");
        assert_eq!(
            Class::containing(seg.len).index(),
            index,
            "filed under wrong class"
        );
        members += 1;
        prev = Some(id);
        cursor = next;
    }
    members
}

fn check_recycled<M: Memory>(a: &TagAllocator<M>) -> u32 {
    let mut recycled = 0;
    let mut cursor = a.link(AT_RECYCLED);
    while let Some(id) = cursor {
        let Record::Recycled { next } = a.record(id) else {
            panic!("live record recycled")
        };
        recycled += 1;
        cursor = next;
    }
    recycled
}

#[test]
fn allocations_grow_the_frontier_in_order() {
    let memory = VecMemory::new();
    let mut a = TagAllocator::create(&memory).unwrap();
    let x = a.allocate(len(10)).unwrap();
    let y = a.allocate(len(5)).unwrap();
    assert_eq!((x.range(), y.range()), (span(0, 10), span(10, 5)));
    assert_eq!(a.frontier(), Slot::new(15));
    check(&a);
}

#[test]
fn release_coalesces_both_neighbours() {
    let memory = VecMemory::new();
    let mut a = TagAllocator::create(&memory).unwrap();
    let tags: Vec<_> = (0..4).map(|_| a.allocate(len(8)).unwrap()).collect();
    let [t0, t1, t2, t3] = <[SpanTag; 4]>::try_from(tags).unwrap();
    a.release(t0);
    a.release(t2);
    assert_eq!(
        check(&a),
        [
            Seen::Free(span(0, 8)),
            Seen::Allocated(span(8, 8)),
            Seen::Free(span(16, 8)),
            Seen::Allocated(span(24, 8)),
        ]
    );
    a.release(t1);
    assert_eq!(
        check(&a),
        [Seen::Free(span(0, 24)), Seen::Allocated(span(24, 8))]
    );
    a.release(t3);
    assert_eq!(check(&a), [Seen::Free(span(0, 32))]);
}

#[test]
fn allocation_splits_the_low_end_and_reuses_records() {
    let memory = VecMemory::new();
    let mut a = TagAllocator::create(&memory).unwrap();
    let big = a.allocate(len(100)).unwrap();
    let fence = a.allocate(len(1)).unwrap();
    a.release(big);
    let small = a.allocate(len(30)).unwrap();
    assert_eq!(small.range(), span(0, 30));
    assert_eq!(
        check(&a),
        [
            Seen::Allocated(span(0, 30)),
            Seen::Free(span(30, 70)),
            Seen::Allocated(span(100, 1))
        ]
    );
    let records = a.record_count();
    a.release(small);
    a.release(fence);
    check(&a);
    a.allocate(len(7)).unwrap();
    assert_eq!(a.record_count(), records, "split reused a recycled record");
    check(&a);
}

#[test]
fn a_free_tail_is_extended_instead_of_abandoned() {
    let memory = VecMemory::new();
    let mut a = TagAllocator::create(&memory).unwrap();
    let keep = a.allocate(len(4)).unwrap();
    let tail = a.allocate(len(10)).unwrap();
    a.release(tail);
    let grown = a.allocate(len(25)).unwrap();
    assert_eq!(grown.range(), span(4, 25));
    assert_eq!(a.frontier(), Slot::new(29));
    assert_eq!(
        check(&a),
        [Seen::Allocated(keep.range()), Seen::Allocated(span(4, 25))]
    );
}

#[test]
fn a_fitting_tail_that_is_not_a_list_head_is_split_not_dropped() {
    let memory = VecMemory::new();
    let mut a = TagAllocator::create(&memory).unwrap();
    // Two free segments in the same class; the tail is not the head of that list.
    let first = a.allocate(len(100)).unwrap();
    let _fence = a.allocate(len(1)).unwrap();
    let tail = a.allocate(len(101)).unwrap();
    a.release(tail);
    a.release(first);
    let taken = a.allocate(len(101)).unwrap();
    assert_eq!(taken.range(), span(101, 101));
    check(&a);
}

#[test]
fn reopen_reads_the_same_state() {
    let memory = VecMemory::new();
    let (kept, before) = {
        let mut a = TagAllocator::create(&memory).unwrap();
        let kept = a.allocate(len(12)).unwrap();
        let gone = a.allocate(len(40)).unwrap();
        a.allocate(len(3)).unwrap();
        a.release(gone);
        (kept, check(&a))
    };
    let kept_id = kept.id();

    let mut b = TagAllocator::open(&memory).unwrap();
    assert_eq!(check(&b), before);
    let recovered = b.recover(kept_id).unwrap();
    assert_eq!(recovered, kept);
    b.release(recovered);
    check(&b);
}

#[test]
fn open_rejects_foreign_memory_and_create_rejects_used_memory() {
    let empty = VecMemory::new();
    assert!(matches!(
        TagAllocator::open(&empty),
        Err(OpenError::NotAnAllocator)
    ));
    let foreign = VecMemory::new();
    foreign.grow(1).unwrap();
    assert!(matches!(
        TagAllocator::open(&foreign),
        Err(OpenError::NotAnAllocator)
    ));
    assert!(matches!(
        TagAllocator::create(&foreign),
        Err(CreateError::NotEmpty)
    ));
}

#[test]
fn recover_rejects_ids_that_are_not_allocated() {
    let memory = VecMemory::new();
    let mut a = TagAllocator::create(&memory).unwrap();
    let x = a.allocate(len(5)).unwrap();
    let id = x.id();
    a.release(x);
    assert_eq!(a.recover(id), Err(NotAllocated));
    assert_eq!(a.recover(TagId::new(99)), Err(NotAllocated));
}

#[test]
fn extend_takes_part_of_a_free_neighbour() {
    let memory = VecMemory::new();
    let mut a = TagAllocator::create(&memory).unwrap();
    let x = a.allocate(len(10)).unwrap();
    let gap = a.allocate(len(20)).unwrap();
    let _fence = a.allocate(len(1)).unwrap();
    a.release(gap);
    let x = a.try_extend(x, len(15)).unwrap();
    assert_eq!(x.range(), span(0, 15));
    assert_eq!(
        check(&a),
        [
            Seen::Allocated(span(0, 15)),
            Seen::Free(span(15, 15)),
            Seen::Allocated(span(30, 1))
        ]
    );
}

#[test]
fn extend_consumes_an_exactly_fitting_neighbour() {
    let memory = VecMemory::new();
    let mut a = TagAllocator::create(&memory).unwrap();
    let x = a.allocate(len(10)).unwrap();
    let gap = a.allocate(len(5)).unwrap();
    let _fence = a.allocate(len(1)).unwrap();
    a.release(gap);
    let x = a.try_extend(x, len(15)).unwrap();
    assert_eq!(
        check(&a),
        [Seen::Allocated(x.range()), Seen::Allocated(span(15, 1))]
    );
}

#[test]
fn extend_at_the_frontier_and_through_a_short_free_tail() {
    let memory = VecMemory::new();
    let mut a = TagAllocator::create(&memory).unwrap();
    let x = a.allocate(len(10)).unwrap();
    let x = a.try_extend(x, len(12)).unwrap();
    assert_eq!((x.range(), a.frontier()), (span(0, 12), Slot::new(12)));
    let tail = a.allocate(len(3)).unwrap();
    a.release(tail);
    let x = a.try_extend(x, len(40)).unwrap();
    assert_eq!((x.range(), a.frontier()), (span(0, 40), Slot::new(40)));
    assert_eq!(check(&a), [Seen::Allocated(span(0, 40))]);
}

#[test]
fn extend_is_blocked_by_an_allocated_or_short_neighbour() {
    let memory = VecMemory::new();
    let mut a = TagAllocator::create(&memory).unwrap();
    let x = a.allocate(len(10)).unwrap();
    let y = a.allocate(len(4)).unwrap();
    let x = a.try_extend(x, len(11)).unwrap_err().into_tag();
    let _fence = a.allocate(len(1)).unwrap();
    a.release(y);
    let before = check(&a);
    let x = a.try_extend(x, len(20)).unwrap_err().into_tag();
    assert_eq!((x.range(), check(&a)), (span(0, 10), before));
    let x = a.try_extend(x, len(10)).unwrap();
    assert_eq!(x.range(), span(0, 10), "not larger: unchanged");
}

/// xorshift64*: deterministic, dependency-free randomness for the stress test.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Mostly small spans with an occasional large one, like leaf spans.
fn random_len(rng: &mut Rng) -> SpanLen {
    let n = if rng.below(10) == 0 {
        rng.below(4_000)
    } else {
        rng.below(64)
    };
    len(u32::try_from(n + 1).unwrap())
}

/// One random allocation, release or in-place extension.
fn random_step<M: Memory>(a: &mut TagAllocator<M>, live: &mut Vec<SpanTag>, rng: &mut Rng) {
    let roll = rng.below(100);
    if live.is_empty() || roll < 45 {
        live.push(a.allocate(random_len(rng)).unwrap());
        return;
    }
    let at = usize::try_from(rng.below(u64::try_from(live.len()).unwrap())).unwrap();
    let tag = live.swap_remove(at);
    if roll < 80 {
        a.release(tag);
    } else {
        let want = len(tag.range().len().get() + random_len(rng).get());
        live.push(a.try_extend(tag, want).unwrap_or_else(Blocked::into_tag));
    }
}

/// The allocated segments are exactly the spans the test holds.
fn assert_allocated_matches<M: Memory>(a: &TagAllocator<M>, live: &[SpanTag]) {
    let mut allocated: Vec<_> = check(a)
        .into_iter()
        .filter_map(|s| match s {
            Seen::Allocated(r) => Some(r),
            Seen::Free(_) => None,
        })
        .collect();
    let mut expected: Vec<_> = live.iter().map(SpanTag::range).collect();
    allocated.sort_by_key(|r| r.start());
    expected.sort_by_key(|r| r.start());
    assert_eq!(allocated, expected);
}

#[test]
fn random_operations_keep_every_invariant() {
    for seed in 1u64..=8 {
        let memory = VecMemory::new();
        let mut a = TagAllocator::create(&memory).unwrap();
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut live = Vec::new();
        for step in 1..=1_500 {
            random_step(&mut a, &mut live, &mut rng);
            if step % 50 == 0 {
                assert_allocated_matches(&a, &live);
            }
        }
        for tag in live.drain(..) {
            a.release(tag);
        }
        assert!(
            check(&a).len() <= 1,
            "everything coalesces back into one free segment"
        );
    }
}
