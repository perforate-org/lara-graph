//! Building a graph from an edge list in one pass.

use super::{Direction, EdgeKind, Error, Graph, LabelBucketKey};
use crate::corrupt::corrupt;
use crate::descriptor::{Descriptor, PAGE_SLOTS, Paged, Run, Slot, TINY_SLOTS, Tiny};
use crate::id::{EdgeRef, Halves, NodeId, NodeState, Position};
use crate::label_bucket;
use crate::label_bucket_list::{self, Lookup};
use crate::memory::Memory;
use crate::page_store::{ByteClass, PageStore};
use crate::paged::{self, Leaf};
use crate::stored::{Index, LabelBucket, Stored, Values};
use crate::targets::{self, BLOCK, Change};
use crate::units::ByteOffset;
use crate::values::{self, Plane};
use alloc::vec::Vec;

/// One half of an input edge before positions are assigned.
struct Half<'a> {
    node: NodeId,
    key: LabelBucketKey,
    target: NodeId,
    value: &'a [u8],
    edge: usize,
    slot: u8,
    position: Position,
}

impl<M: Memory> Graph<M> {
    /// Builds the graph from `edges` in one pass. The graph must have its nodes and width
    /// declarations already, and no edges; otherwise this returns [`Error::NotEmpty`].
    ///
    /// Within a bucket, edges get positions in input order, like [`Placement::Append`]. The
    /// `placement` of each [`NewEdge`] is ignored because there are no holes to reuse.
    ///
    /// # Errors
    ///
    /// [`Error::UnknownNode`], [`Error::Detaching`], [`Error::Deleted`],
    /// [`Error::WrongValueLength`], [`Error::TooManyLabelBuckets`], [`Error::AddressSpaceExhausted`],
    /// [`Error::OutOfMemory`], or [`Error::NotEmpty`].
    #[expect(
        clippy::too_many_lines,
        clippy::indexing_slicing,
        reason = "one bulk pass owns these bounds"
    )]
    pub fn bulk_load(&mut self, edges: &[super::NewEdge<'_>]) -> Result<Vec<Halves>, Error> {
        for v in 0..self.node_count() {
            if self.label_bucket_list(NodeId::new(v))?.is_some() {
                return Err(Error::NotEmpty);
            }
        }
        for edge in edges {
            self.check(edge.from())?;
            self.check(edge.to())?;
            for v in [edge.from(), edge.to()] {
                match self.record(v)?.state {
                    NodeState::Live => {}
                    NodeState::Detaching => return Err(Error::Detaching(v)),
                    NodeState::Deleted => return Err(Error::Deleted(v)),
                }
            }
            self.value_width(edge.label(), edge.value())?;
        }
        if edges.is_empty() {
            return Ok(Vec::new());
        }

        let mut positions = alloc::vec![[None; 2]; edges.len()];
        let mut halves: Vec<Half<'_>> = Vec::with_capacity(edges.len() * 2);
        for (edge_index, edge) in edges.iter().enumerate() {
            match edge.kind() {
                EdgeKind::Directed => {
                    let out = Half {
                        node: edge.from(),
                        key: LabelBucketKey {
                            label: edge.label(),
                            direction: Direction::Outgoing,
                        },
                        target: edge.to(),
                        value: edge.value(),
                        edge: edge_index,
                        slot: 0,
                        position: Position::new(0),
                    };
                    let into = Half {
                        node: edge.to(),
                        key: LabelBucketKey {
                            label: edge.label(),
                            direction: Direction::Incoming,
                        },
                        target: edge.from(),
                        value: edge.value(),
                        edge: edge_index,
                        slot: 1,
                        position: Position::new(0),
                    };
                    halves.push(out);
                    halves.push(into);
                }
                EdgeKind::Undirected => {
                    let (low, high) = (edge.from().min(edge.to()), edge.from().max(edge.to()));
                    let high_half = Half {
                        node: high,
                        key: LabelBucketKey {
                            label: edge.label(),
                            direction: Direction::Undirected,
                        },
                        target: low,
                        value: edge.value(),
                        edge: edge_index,
                        slot: 0,
                        position: Position::new(0),
                    };
                    if low == high {
                        halves.push(high_half);
                    } else {
                        let low_half = Half {
                            node: low,
                            key: LabelBucketKey {
                                label: edge.label(),
                                direction: Direction::Undirected,
                            },
                            target: high,
                            value: edge.value(),
                            edge: edge_index,
                            slot: 1,
                            position: Position::new(0),
                        };
                        halves.push(high_half);
                        halves.push(low_half);
                    }
                }
            }
        }
        halves.sort_by_key(|half| (half.node, half.key));

        let mut start = 0;
        while start < halves.len() {
            let mut end = start + 1;
            while end < halves.len()
                && halves[end].node == halves[start].node
                && halves[end].key == halves[start].key
            {
                end += 1;
            }
            for (offset, half) in halves[start..end].iter_mut().enumerate() {
                let position = Position::new(u32::try_from(offset).unwrap_or(u32::MAX));
                half.position = position;
                positions[half.edge][usize::from(half.slot)] = Some(position);
            }
            start = end;
        }
        for edge in edges {
            let index = positions.iter().position(|_| false).unwrap_or_default();
            let _ = index;
            let _ = edge;
        }

        let mut list: Option<label_bucket_list::LabelBucketList> = None;
        let mut current: Option<NodeId> = None;
        let mut start = 0;
        while start < halves.len() {
            let node = halves[start].node;
            if current != Some(node) {
                if let (Some(previous), Some(built)) = (current, list) {
                    self.set_label_bucket_list(previous, Some(built));
                }
                current = Some(node);
                list = None;
            }
            let mut end = start + 1;
            while end < halves.len()
                && halves[end].node == node
                && halves[end].key == halves[start].key
            {
                end += 1;
            }
            let stored = self.build_bucket(&halves[start..end])?;
            let pages = self.label_bucket_lists.pages();
            let gap = match label_bucket_list::lookup(&pages, list, halves[start].key) {
                Lookup::Present { .. } => {
                    return Err(Error::TooManyLabelBuckets(node));
                }
                Lookup::Absent { gap } => gap,
            };
            self.label_bucket_lists
                .reserve(label_bucket_list::PAGES_PER_ADD)?;
            list = Some(label_bucket_list::add(
                &mut self.label_bucket_lists,
                list,
                gap,
                halves[start].key,
                &stored,
            )?);
            start = end;
        }
        if let (Some(node), Some(built)) = (current, list) {
            self.set_label_bucket_list(node, Some(built));
        }

        for (edge_index, edge) in edges.iter().enumerate() {
            if edge.kind() == EdgeKind::Undirected && edge.from() == edge.to() {
                positions[edge_index][1] = positions[edge_index][0];
            }
        }

        let mut out = Vec::with_capacity(edges.len());
        for (edge_index, edge) in edges.iter().enumerate() {
            let first = positions[edge_index][0]
                .unwrap_or_else(|| corrupt("a bulk edge without its canonical half"));
            let second = positions[edge_index][1]
                .unwrap_or_else(|| corrupt("a bulk edge without its other half"));
            let (which, other) = match edge.kind() {
                EdgeKind::Directed => (
                    EdgeRef::new(edge.from(), edge.label(), Direction::Outgoing, first),
                    EdgeRef::new(edge.to(), edge.label(), Direction::Incoming, second),
                ),
                EdgeKind::Undirected => {
                    let (low, high) = (edge.from().min(edge.to()), edge.from().max(edge.to()));
                    (
                        EdgeRef::new(high, edge.label(), Direction::Undirected, first),
                        EdgeRef::new(low, edge.label(), Direction::Undirected, second),
                    )
                }
            };
            out.push(Halves::new(which, other));
        }
        Ok(out)
    }

    #[expect(clippy::indexing_slicing, reason = "the group is non-empty")]
    fn build_bucket(&mut self, halves: &[Half<'_>]) -> Result<Stored, Error> {
        let label = halves[0].key.label;
        let width = self.widths.get(label);
        let count = u32::try_from(halves.len()).map_err(|_| Error::AddressSpaceExhausted)?;
        let tiny_slots = u32::try_from(TINY_SLOTS).unwrap_or(3);
        if width.is_none() && count <= tiny_slots {
            let mut tiny = Tiny::EMPTY;
            for half in halves {
                tiny = tiny.push(Slot::Live(half.target));
            }
            return Ok(LabelBucket::plain(Descriptor::Tiny(tiny)).stored());
        }
        let mut bucket = if count <= PAGE_SLOTS {
            let extent = self.edges.allocate(label_bucket::run_class(count))?;
            write_halves(&self.edges, extent.offset(), halves);
            LabelBucket::plain(Descriptor::Run(Run {
                extent,
                stored: label_bucket::narrow(count),
                live: label_bucket::narrow(count),
            }))
        } else {
            let paged = self.build_paged(halves)?;
            LabelBucket::plain(Descriptor::Paged(paged))
        };
        if let Some(width) = width {
            let mut plane = Plane::Empty;
            for (position, half) in halves.iter().enumerate() {
                plane = values::write(
                    &mut self.edges,
                    plane,
                    width,
                    u32::try_from(position).map_err(|_| Error::AddressSpaceExhausted)?,
                    half.value,
                )?;
            }
            bucket.values = Some(Values { plane, width });
        }
        if matches!(bucket.slots, Descriptor::Paged(_)) {
            let end = count / BLOCK * BLOCK;
            if end > 0 {
                let changes: Vec<Change> = (0..end)
                    .map(|at| Change {
                        key: (
                            halves[usize::try_from(at).unwrap_or(0)].target.index(),
                            at / BLOCK,
                        ),
                        delta: 1,
                    })
                    .collect();
                let plan = targets::plan(&self.edges.pages(), None, &changes);
                self.edges.reserve(plan.pages())?;
                bucket.index = Index::of(targets::commit(&mut self.edges, plan)?);
            }
        }
        if bucket.extended() {
            bucket.header = Some(self.edges.allocate(crate::stored::HEADER_CLASS)?);
            bucket.write_header(&self.edges);
        }
        Ok(bucket.stored())
    }

    #[expect(
        clippy::indexing_slicing,
        reason = "page ranges come from the slot count"
    )]
    fn build_paged(&mut self, halves: &[Half<'_>]) -> Result<Paged, Error> {
        let count = u32::try_from(halves.len()).map_err(|_| Error::AddressSpaceExhausted)?;
        let page_slots = usize::try_from(PAGE_SLOTS).unwrap_or(1024);
        let first = self.edges.allocate(ByteClass::PAGE)?.page();
        let (mut index, second) =
            paged::start(&mut self.edges, Leaf::data(first, 0, paged::NO_HOLE))?;
        write_halves(&self.edges, first.offset(), &halves[..page_slots]);
        write_halves(
            &self.edges,
            second.offset(),
            &halves
                [page_slots..usize::try_from(count.min(2 * PAGE_SLOTS)).unwrap_or(2 * page_slots)],
        );
        let page_count = paged::data_pages(count);
        for page in 2..page_count {
            let before = page * PAGE_SLOTS;
            let paged = Paged {
                index,
                stored: before,
                live: before,
            };
            let (next, data) = paged::append_page(&mut self.edges, &paged)?;
            write_halves(
                &self.edges,
                data.offset(),
                &halves[usize::try_from(before).unwrap_or(0)
                    ..usize::try_from(count.min(before + PAGE_SLOTS)).unwrap_or(0)],
            );
            index = next;
        }
        Ok(Paged {
            index,
            stored: count,
            live: count,
        })
    }
}

#[expect(
    clippy::indexing_slicing,
    reason = "one page of slots bounds the slice"
)]
fn write_halves<M: Memory>(store: &PageStore<M>, at: ByteOffset, halves: &[Half<'_>]) {
    let mut bytes = [0u8; 4096];
    let len = halves.len() * 4;
    for (dst, half) in bytes[..len].as_chunks_mut::<4>().0.iter_mut().zip(halves) {
        *dst = Slot::Live(half.target).encode();
    }
    store.write(at, &bytes[..len]);
}
