//! Probe seam for byte-traffic observation (extracted from the
//! traffic-compare Probe implementation; new file, sealed product untouched).
//!
//! [`Probe`] wraps a [`Memory`] backend and records per-call traffic.
//! [`RegionTraffic`] snapshots named probe handles into per-region totals.
//!
//! Distinct-byte sets use a `Vec<u64>` bitmap (1 bit per byte) with word-level
//! OR updates. `track_distinct` disables bitmap maintenance when only basic
//! call/byte counts are needed.

use lara_graph::memory::{GrowFailed, Memory};
use std::cell::RefCell;
use std::rc::Rc;

/// Observation of one region during a measured phase: call counts and byte
/// totals per operation, plus distinct bytes touched (overlap-counting union).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Traffic {
    pub reads: u64,
    pub read_bytes: u64,
    pub mapped_reads: u64,
    pub mapped_read_bytes: u64,
    pub writes: u64,
    pub write_bytes: u64,
    pub distinct_read_bytes: u64,
    pub distinct_write_bytes: u64,
    pub grows: u64,
}

impl Traffic {
    fn add(&mut self, other: &Traffic) {
        self.reads += other.reads;
        self.read_bytes += other.read_bytes;
        self.mapped_reads += other.mapped_reads;
        self.mapped_read_bytes += other.mapped_read_bytes;
        self.writes += other.writes;
        self.write_bytes += other.write_bytes;
        self.distinct_read_bytes += other.distinct_read_bytes;
        self.distinct_write_bytes += other.distinct_write_bytes;
        self.grows += other.grows;
    }
}

fn count_bitmap(bits: &[u64]) -> u64 {
    bits.iter().map(|w| w.count_ones() as u64).sum()
}

/// Shared probe state: counters plus optional distinct bitmaps.
struct ProbeState {
    counters: Traffic,
    touched_reads: Vec<u64>,
    touched_writes: Vec<u64>,
    track_distinct: bool,
}

/// A `Memory` wrapper that records per-call traffic. Forwarding only; the
/// graph never sees the bookkeeping. Handles are cheap to clone and share
/// the same counters, so the harness can retain clones while the graph owns
/// the originals. Distinct-byte sets span since the last `reset` (one phase).
pub struct Probe<Inner> {
    inner: Inner,
    state: Rc<RefCell<ProbeState>>,
}

impl<Inner> Probe<Inner> {
    /// A handle sharing this probe's counters without touching `inner`
    /// (the graph keeps owning the backend).
    pub fn handle(&self) -> ProbeHandle {
        ProbeHandle {
            state: Rc::clone(&self.state),
        }
    }
}

/// Shared-counter view of a [`Probe`]: traffic snapshots and phase resets
/// without owning the backend.
#[derive(Clone)]
pub struct ProbeHandle {
    state: Rc<RefCell<ProbeState>>,
}

impl ProbeHandle {
    /// One region's snapshot.
    pub fn traffic(&self) -> Traffic {
        let st = self.state.borrow();
        let mut t = st.counters;
        t.distinct_read_bytes = count_bitmap(&st.touched_reads);
        t.distinct_write_bytes = count_bitmap(&st.touched_writes);
        t
    }

    /// Starts a new observation phase: zero counters and distinct sets.
    pub fn reset(&self) {
        let mut st = self.state.borrow_mut();
        st.counters = Traffic::default();
        st.touched_reads.clear();
        st.touched_writes.clear();
    }
}

impl<Inner: Memory> Probe<Inner> {
    pub fn new(inner: Inner) -> Self {
        Self::with_distinct(inner, true)
    }

    /// Creates a probe with distinct-bitmap tracking enabled or disabled.
    /// Disabled probes skip all bitmap maintenance; basic call/byte counts
    /// and `distinct_*` stays zero.
    pub fn with_distinct(inner: Inner, track_distinct: bool) -> Self {
        Self {
            inner,
            state: Rc::new(RefCell::new(ProbeState {
                counters: Traffic::default(),
                touched_reads: Vec::new(),
                touched_writes: Vec::new(),
                track_distinct,
            })),
        }
    }

    /// One region's snapshot.
    pub fn traffic(&self) -> Traffic {
        self.handle().traffic()
    }

    /// Starts a new observation phase: zero counters and distinct sets.
    pub fn reset(&self) {
        self.handle().reset();
    }
}

impl<Inner: Memory> Memory for Probe<Inner> {
    fn read(&self, offset: u64, buf: &mut [u8]) {
        self.inner.read(offset, buf);
        let track = self.state.borrow().track_distinct;
        {
            let mut st = self.state.borrow_mut();
            st.counters.reads += 1;
            st.counters.read_bytes += buf.len() as u64;
        }
        mark_bitmap_read(&self.state, track, offset, buf.len());
    }

    fn read_mapped(&self, offset: u64, buf: &mut [u8]) {
        self.inner.read_mapped(offset, buf);
        let track = self.state.borrow().track_distinct;
        {
            let mut st = self.state.borrow_mut();
            st.counters.mapped_reads += 1;
            st.counters.mapped_read_bytes += buf.len() as u64;
        }
        // read_mapped copies bytes, so they belong in the distinct set too.
        mark_bitmap_read(&self.state, track, offset, buf.len());
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        self.inner.write(offset, buf);
        let track = self.state.borrow().track_distinct;
        {
            let mut st = self.state.borrow_mut();
            st.counters.writes += 1;
            st.counters.write_bytes += buf.len() as u64;
        }
        if track {
            mark_writes(&self.state, offset, buf.len());
        }
    }

    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        self.inner.grow(pages).inspect(|_| {
            self.state.borrow_mut().counters.grows += 1;
        })
    }

    fn fence(&self) {
        self.inner.fence()
    }
}

/// Marks the read bitmap. Split from [`Memory::read`] to keep borrow scopes small.
fn mark_bitmap_read(state: &Rc<RefCell<ProbeState>>, track: bool, offset: u64, len: usize) {
    if !track || len == 0 {
        return;
    }
    let binding = Rc::clone(state);
    let mut st = binding.borrow_mut();
    let start = offset as usize;
    let end = start + len;
    let words = end.div_ceil(64);
    if st.touched_reads.len() < words {
        st.touched_reads.resize(words, 0);
    }
    let s = start / 64;
    let e = end.div_ceil(64);
    if s + 1 >= e {
        let lo = start % 64;
        let mask = if len >= 64 {
            u64::MAX
        } else {
            ((1u64 << len) - 1) << lo
        };
        st.touched_reads[s] |= mask;
        return;
    }
    let lo = start % 64;
    if lo != 0 {
        st.touched_reads[s] |= u64::MAX << lo;
    } else {
        st.touched_reads[s] |= u64::MAX;
    }
    for w in &mut st.touched_reads[s + 1..e - 1] {
        *w = u64::MAX;
    }
    let rem = end % 64;
    if rem != 0 {
        st.touched_reads[e - 1] |= (1u64 << rem) - 1;
    } else if e - 1 > s {
        st.touched_reads[e - 1] = u64::MAX;
    }
}

/// Marks the write bitmap.
fn mark_writes(state: &Rc<RefCell<ProbeState>>, offset: u64, len: usize) {
    if len == 0 {
        return;
    }
    let mut st = state.borrow_mut();
    let start = offset as usize;
    let end = start + len;
    let words = end.div_ceil(64);
    if st.touched_writes.len() < words {
        st.touched_writes.resize(words, 0);
    }
    let s = start / 64;
    let e = end.div_ceil(64);
    if s + 1 >= e {
        let lo = start % 64;
        let mask = if len >= 64 {
            u64::MAX
        } else {
            ((1u64 << len) - 1) << lo
        };
        st.touched_writes[s] |= mask;
        return;
    }
    let lo = start % 64;
    if lo != 0 {
        st.touched_writes[s] |= u64::MAX << lo;
    } else {
        st.touched_writes[s] |= u64::MAX;
    }
    for w in &mut st.touched_writes[s + 1..e - 1] {
        *w = u64::MAX;
    }
    let rem = end % 64;
    if rem != 0 {
        st.touched_writes[e - 1] |= (1u64 << rem) - 1;
    } else if e - 1 > s {
        st.touched_writes[e - 1] = u64::MAX;
    }
}

/// Final observation for one arm: one entry per region plus their sum.
#[derive(Clone, Debug, Default)]
pub struct RegionTraffic {
    /// One `Traffic` per memory region (LARA: 8, DGAP: 1).
    pub regions: Vec<(&'static str, Traffic)>,
    /// Sum over all regions.
    pub total: Traffic,
}

impl RegionTraffic {
    /// Snapshots the given named probe handles (shared with the graph-owned
    /// probes) and sums them into `total`.
    pub fn from_handles(entries: &[(&'static str, &ProbeHandle)]) -> Self {
        let mut total = Traffic::default();
        let regions: Vec<(&'static str, Traffic)> = entries
            .iter()
            .map(|(name, handle)| {
                let t = handle.traffic();
                total.add(&t);
                (*name, t)
            })
            .collect();
        Self { regions, total }
    }
}

/// Generic bridge: lets any [`Memory`] backend drive the dgap crate's
/// [`dgap::Memory`] trait through an optional [`Probe`].
pub struct ProbedDgapBridge<M>(pub M);

impl<M: Memory> dgap::Memory for ProbedDgapBridge<M> {
    fn read(&self, offset: u64, buf: &mut [u8]) {
        lara_graph::Memory::read(&self.0, offset, buf)
    }
    fn write(&mut self, offset: u64, buf: &[u8]) {
        lara_graph::Memory::write(&self.0, offset, buf)
    }
    fn size(&self) -> u64 {
        lara_graph::Memory::size(&self.0) * lara_graph::memory::PAGE_SIZE
    }
    fn grow(&mut self, pages: u64) -> Result<u64, dgap::GrowFailed> {
        lara_graph::Memory::grow(&self.0, pages)
            .map(|new_pages| new_pages * lara_graph::memory::PAGE_SIZE)
            .map_err(|e| dgap::GrowFailed {
                current_size: e.current_pages * lara_graph::memory::PAGE_SIZE,
                delta_pages: e.delta_pages,
            })
    }
    fn fence(&self) {
        self.0.fence()
    }
}
