//! Shared strict real-backing observer; construction and oracles are outside intervals.
use super::{GrowFailed, LaraMemories, Memory, PAGE_SIZE, VecMemory};
use std::{cell::RefCell, fmt::Write as _, rc::Rc};

#[derive(Debug, Default)]
pub struct Trace {
    pub reads: Vec<(u64, usize, bool)>,
    pub writes: Vec<(u64, Vec<u8>)>,
    pub grows: Vec<u64>,
}
#[derive(Clone, Default)]
pub struct Probe {
    pub mem: VecMemory,
    trace: Rc<RefCell<Option<Trace>>>,
}
impl Probe {
    fn reading(&self, off: u64, len: usize, mapped: bool) {
        assert!(off + len as u64 <= self.mem.size() * PAGE_SIZE);
        if let Some(t) = self.trace.borrow_mut().as_mut() {
            t.reads.push((off, len, mapped));
        }
    }
    pub fn bytes(&self) -> Vec<u8> {
        let mut b = vec![0; (self.mem.size() * PAGE_SIZE) as usize];
        self.mem.read(0, &mut b);
        b
    }
}
impl Memory for Probe {
    fn read(&self, off: u64, b: &mut [u8]) {
        self.reading(off, b.len(), false);
        self.mem.read(off, b);
    }
    fn read_mapped(&self, off: u64, b: &mut [u8]) {
        self.reading(off, b.len(), true);
        self.mem.read_mapped(off, b);
    }
    fn write(&self, off: u64, b: &[u8]) {
        assert!(off + b.len() as u64 <= self.mem.size() * PAGE_SIZE);
        if let Some(t) = self.trace.borrow_mut().as_mut() {
            t.writes.push((off, b.to_vec()));
        }
        self.mem.write(off, b);
    }
    fn grow(&self, n: u64) -> Result<u64, GrowFailed> {
        if let Some(t) = self.trace.borrow_mut().as_mut() {
            t.grows.push(n);
        }
        self.mem.grow(n)
    }
    fn size(&self) -> u64 {
        self.mem.size()
    }
    fn fence(&self) {
        self.mem.fence();
    }
}
pub fn regions() -> [Probe; 8] {
    std::array::from_fn(|_| Probe::default())
}
pub fn bundle(r: &[Probe; 8]) -> LaraMemories<Probe> {
    LaraMemories {
        vertices: r[0].clone(),
        edges: r[1].clone(),
        logs: r[2].clone(),
        counts: r[3].clone(),
        spans: r[4].clone(),
        free_spans: r[5].clone(),
        blocks: r[6].clone(),
        buckets: r[7].clone(),
    }
}
/// Captures a public operation silently, freezing every region before returning.
pub fn capture<T>(r: &[Probe; 8], op: impl FnOnce() -> T) -> (T, [Trace; 8]) {
    for p in r {
        assert!(p.trace.replace(Some(Trace::default())).is_none());
    }
    let result = op();
    let ts = std::array::from_fn(|i| r[i].trace.take().unwrap());
    (result, ts)
}
pub fn observe<T>(tag: &str, r: &[Probe; 8], op: impl FnOnce() -> T) -> (T, [Trace; 8]) {
    let (result, ts) = capture(r, op);
    for (i, t) in ts.iter().enumerate() {
        let region = [
            "vertices",
            "edges",
            "logs",
            "counts",
            "spans",
            "free_spans",
            "blocks",
            "buckets",
        ][i];
        let mapped = t.reads.iter().filter(|r| r.2).count();
        println!(
            "OBS,{tag},{region},{},{mapped},{},{},{},{},{},{}",
            t.reads.len() - mapped,
            t.reads.iter().map(|r| r.1).sum::<usize>(),
            t.writes.len(),
            t.writes.iter().map(|w| w.1.len()).sum::<usize>(),
            t.grows.len(),
            r[i].size(),
            r[i].mem.capacity_bytes()
        );
        for (k, (off, n, m)) in t.reads.iter().enumerate() {
            println!("READ,{tag},{region},{k},{off},{n},{m}");
        }
        for (k, (off, b)) in t.writes.iter().enumerate() {
            let mut hex = String::new();
            for x in b {
                write!(hex, "{x:02x}").unwrap();
            }
            println!("WRITE,{tag},{region},{k},{off},{},{hex}", b.len());
        }
        for (k, n) in t.grows.iter().enumerate() {
            println!("GROW,{tag},{region},{k},{n}");
        }
        let hash = r[i].bytes().iter().fold(0xcbf29ce484222325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        println!("STATE,{tag},{region},{hash:016x}");
    }
    (result, ts)
}
