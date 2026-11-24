//! A graph's lifecycle: declared widths and edges surviving a reopen, `init`, `new` and `load`
//! on foreign or other-version memories, and ic-stable-structures' memories.

use super::*;

#[test]
fn label_widths_are_declared_once_and_survive_a_reopen() {
    let backing = Backing::default();
    let w = |bytes| Width::new(bytes).unwrap();
    {
        let mut g = backing.create(1);
        assert_eq!(g.width(l(3)), None);
        g.declare_width(l(3), w(8)).unwrap();
        g.declare_width(l(1), w(4096)).unwrap();
        g.declare_width(l(3), w(8)).unwrap();
        assert_eq!(
            g.declare_width(l(3), w(16)),
            Err(Error::WidthConflict {
                label: l(3),
                declared: w(8)
            })
        );
        check(&g);
    }
    let g = backing.open();
    assert_eq!(
        (g.width(l(1)), g.width(l(3)), g.width(l(2))),
        (Some(w(4096)), Some(w(8)), None)
    );
    check(&g);
    assert_eq!(Width::new(0), None);
    assert_eq!(Width::new(4097), None);
}

/// The edges of the first four nodes, under labels 0 to 2, on both sides.
fn snapshot(g: &Graph<VecMemory>) -> Vec<Vec<Vec<(u32, u32)>>> {
    (0..4.min(g.node_count()))
        .map(|x| {
            (0..3)
                .flat_map(|y| {
                    [Direction::Outgoing, Direction::Incoming].map(|s| edges_on(g, v(x), l(y), s))
                })
                .collect()
        })
        .collect()
}

#[test]
fn reopen_reads_the_same_graph() {
    let backing = Backing::default();
    let before = {
        let mut g = backing.create(4);
        for i in 0..3_000 {
            insert(
                &mut g,
                v(i % 4),
                l(u16::try_from(i % 3).unwrap()),
                v((i * 7) % 4),
            );
        }
        remove(&mut g, v(1), l(1), p(3));
        snapshot(&g)
    };
    let mut g = backing.open();
    assert_eq!(snapshot(&g), before);
    check(&g);
    insert(&mut g, v(2), l(1), v(1));
    check(&g);
}

/// `init` formats empty or foreign memories and reopens a graph; `load` refuses what is not a
/// graph or is another version; `new` formats over a graph; `into_memories` gives them back as
/// they were left.
#[test]
fn init_new_and_load_follow_ic_stable_structures() {
    let backing = Backing::default();
    assert!(matches!(
        Graph::try_load(backing.memories()),
        Err(InitError::NotAGraph)
    ));
    let mut g = Graph::init(backing.memories());
    for _ in 0..3 {
        g.add_node().unwrap();
    }
    insert(&mut g, v(0), l(1), v(2));
    let before = snapshot(&g);
    let memories = g.into_memories();
    let reopened = Graph::init(memories);
    assert_eq!(snapshot(&reopened), before);
    check(&reopened);
    // Formatting over a graph leaves an empty one.
    let fresh = Graph::new(backing.memories());
    assert_eq!(fresh.node_count(), 0);
    check(&fresh);
    // A foreign magic in the nodes memory: init formats, as ic-stable-structures' BTreeMap does.
    let foreign = Backing::default();
    foreign.memories[0].grow(1).unwrap();
    foreign.memories[0].write(0, b"XYZ\x01");
    assert_eq!(Graph::init(foreign.memories()).node_count(), 0);
    // Another layout version is refused by both.
    backing.memories[0].write(3, &[2]);
    for result in [
        Graph::try_load(backing.memories()),
        Graph::try_init(backing.memories()),
    ] {
        assert!(matches!(result, Err(InitError::UnsupportedVersion(2))));
    }
    // A graph's nodes memory without its stores is not a graph.
    let partial = Backing::default();
    drop(Graph::new(partial.memories()));
    partial.memories[1].write(0, b"XYZ");
    assert!(matches!(
        Graph::try_init(partial.memories()),
        Err(InitError::NotAGraph)
    ));
}

/// ic-stable-structures' memories serve as a graph's, through a memory manager, and a graph
/// reopens from them with `init`.
#[cfg(feature = "ic-stable-structures")]
#[test]
fn ic_stable_structures_memories_hold_a_graph() {
    use crate::memories::MemoryRole;
    use ic_stable_structures::DefaultMemoryImpl;
    use ic_stable_structures::memory_manager::{MemoryId, MemoryManager};
    let manager = MemoryManager::init(DefaultMemoryImpl::default());
    let memories = || Memories::from_fn(|role| manager.get(MemoryId::new(role.index())));
    let mut g = Graph::init(memories());
    let (a, b) = (g.add_node().unwrap(), g.add_node().unwrap());
    let halves = g
        .add_edge(NewEdge::directed(a, l(1), b, Placement::Append, &[]))
        .unwrap();
    drop(g);
    let g = Graph::init(memories());
    assert_eq!(g.node_count(), 2);
    assert_eq!(g.counterpart(halves.canonical()), Ok(Some(halves.other())));
    let nodes = manager.get(MemoryId::new(MemoryRole::Nodes.index()));
    assert!(ic_stable_structures::Memory::size(&nodes) > 0);
}
