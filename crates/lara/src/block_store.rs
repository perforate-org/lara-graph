//! LTB raw block store: fixed 4 KiB payload blocks behind [`Memory`].
//!
//! Simplified port of `ic-stable-lara/src/labeled/ltb_raw_block_store.rs`:
//! 16-byte block headers (kind + owner + level) and 4096-byte payloads.
//! Blocks are dense; free blocks form an intrusive free list for reuse.

use crate::memory::{GrowFailed, Memory, safe_write};

/// Payload bytes per block.
pub const BLOCK_PAYLOAD_BYTES: usize = 4096;
/// Header bytes per block.
pub const BLOCK_HEADER_BYTES: usize = 16;
/// Full block stride.
pub const BLOCK_STRIDE: u64 = BLOCK_HEADER_BYTES as u64 + BLOCK_PAYLOAD_BYTES as u64;

const HEADER_SIZE: u64 = 64;
const OFFSET_TAIL_NEXT: u64 = 8;
const OFFSET_FREE_HEAD: u64 = 16;
const OFFSET_FREE_COUNT: u64 = 20;

const MAGIC: [u8; 3] = *b"LTB";
const NULL_BLOCK: u32 = u32::MAX;

/// Block kinds (1-byte headers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockKind {
    /// On the free list.
    Free = 0,
    /// Data block (leaf).
    Edge = 1,
    /// Interior (index) block.
    EdgeInterior = 3,
}

impl BlockKind {
    #[allow(dead_code, reason = "kind decoding lands with persisted reopening")]
    fn from_byte(b: u8) -> Self {
        match b {
            0 => Self::Free,
            1 => Self::Edge,
            3 => Self::EdgeInterior,
            _ => panic!("LtbRawBlockStore: unknown block kind {b}"),
        }
    }
}

/// Errors raised by block access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockError {
    /// Block id was never minted or already released.
    NotMinted {
        /// The offending block id.
        id: u32,
    },
    /// Payload write exceeds the 4096-byte capacity.
    PayloadOverflow {
        /// The offending block id.
        id: u32,
        /// Requested write length in bytes.
        len: usize,
    },
    /// No more blocks can be minted.
    OutOfBlocks,
}

impl core::fmt::Display for BlockError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotMinted { id } => write!(f, "block {id} not minted"),
            Self::PayloadOverflow { id, len } => {
                write!(f, "block {id} payload write of {len} bytes overflows")
            }
            Self::OutOfBlocks => write!(f, "block store exhausted"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BlockError {}

/// Dense block store over `Memory`.
pub struct LtbRawBlockStore<M: Memory> {
    memory: M,
    /// First never-minted block id.
    tail_next: u32,
    /// Free-list head.
    free_head: u32,
    /// Number of blocks on the free list.
    free_count: u32,
}

impl<M: Memory> LtbRawBlockStore<M> {
    /// Creates a fresh store over `memory` (writes the 64-byte header).
    pub fn new(memory: M) -> Result<Self, GrowFailed> {
        safe_write(&memory, HEADER_SIZE - 1, &[0u8])?;
        safe_write(&memory, 0, &MAGIC)?;
        let s = Self {
            memory,
            tail_next: 0,
            free_head: NULL_BLOCK,
            free_count: 0,
        };
        s.persist_header()?;
        Ok(s)
    }

    fn persist_header(&self) -> Result<(), GrowFailed> {
        safe_write(
            &self.memory,
            OFFSET_TAIL_NEXT,
            &self.tail_next.to_le_bytes(),
        )?;
        safe_write(
            &self.memory,
            OFFSET_FREE_HEAD,
            &self.free_head.to_le_bytes(),
        )?;
        safe_write(
            &self.memory,
            OFFSET_FREE_COUNT,
            &self.free_count.to_le_bytes(),
        )?;
        Ok(())
    }

    /// Number of blocks minted (ever).
    pub fn minted(&self) -> u32 {
        self.tail_next
    }

    fn block_offset(id: u32) -> u64 {
        HEADER_SIZE + id as u64 * BLOCK_STRIDE
    }

    fn mint(&mut self) -> Result<u32, BlockError> {
        let id = if self.free_head != NULL_BLOCK {
            let id = self.free_head;
            // Read next-free from the block header (owner field).
            let mut buf = [0u8; 4];
            self.memory.read(Self::block_offset(id) + 4, &mut buf);
            self.free_head = u32::from_le_bytes(buf);
            self.free_count -= 1;
            id
        } else {
            let id = self.tail_next;
            self.tail_next = self
                .tail_next
                .checked_add(1)
                .ok_or(BlockError::OutOfBlocks)?;
            id
        };
        self.write_block_header(id, BlockKind::Edge, 0)
            .map_err(|_| BlockError::OutOfBlocks)?;
        self.persist_header().map_err(|_| BlockError::OutOfBlocks)?;
        Ok(id)
    }

    fn write_block_header(
        &mut self,
        id: u32,
        kind: BlockKind,
        level: u8,
    ) -> Result<(), GrowFailed> {
        let mut hdr = [0u8; BLOCK_HEADER_BYTES];
        hdr[0] = kind as u8;
        // Reserve the payload too: Memory::write need not grow implicitly.
        safe_write(
            &self.memory,
            Self::block_offset(id) + BLOCK_STRIDE - 1,
            &[0],
        )?;
        self.memory.write(Self::block_offset(id), &hdr);
        let _ = level;
        Ok(())
    }

    /// Reads one block's full 4096-byte payload into `dst`.
    pub fn read_payload(
        &self,
        id: u32,
        dst: &mut [u8; BLOCK_PAYLOAD_BYTES],
    ) -> Result<(), BlockError> {
        if id >= self.tail_next {
            return Err(BlockError::NotMinted { id });
        }
        self.memory
            .read(Self::block_offset(id) + BLOCK_HEADER_BYTES as u64, dst);
        Ok(())
    }

    /// Reads one block's full payload without the trailing zero-fill
    /// (tree-scan fast path). The caller guarantees the block is minted
    /// and fully mapped; see [`crate::memory::Memory::read_mapped`].
    /// Only the scan path uses this — all other readers keep the
    /// zero-fill contract of [`LtbRawBlockStore::read_payload`].
    pub fn read_payload_mapped(
        &self,
        id: u32,
        dst: &mut [u8; BLOCK_PAYLOAD_BYTES],
    ) -> Result<(), BlockError> {
        if id >= self.tail_next {
            return Err(BlockError::NotMinted { id });
        }
        self.memory
            .read_mapped(Self::block_offset(id) + BLOCK_HEADER_BYTES as u64, dst);
        Ok(())
    }

    /// Writes one block's full payload from `src` (mints implicitly).
    pub fn write_payload(
        &mut self,
        id: u32,
        src: &[u8; BLOCK_PAYLOAD_BYTES],
    ) -> Result<(), BlockError> {
        if id >= self.tail_next {
            return Err(BlockError::NotMinted { id });
        }
        self.memory
            .write(Self::block_offset(id) + BLOCK_HEADER_BYTES as u64, src);
        Ok(())
    }

    /// Mints a zeroed block and returns its id (convenience).
    pub fn mint_zeroed(&mut self) -> Result<u32, BlockError> {
        let id = self.mint()?;
        let zero = [0u8; BLOCK_PAYLOAD_BYTES];
        self.write_payload(id, &zero)?;
        Ok(id)
    }

    /// Reads `dst.len()` bytes at `offset` within block `id`'s payload.
    pub fn read_payload_partial(
        &self,
        id: u32,
        offset: usize,
        dst: &mut [u8],
    ) -> Result<(), BlockError> {
        if id >= self.tail_next {
            return Err(BlockError::NotMinted { id });
        }
        if offset + dst.len() > BLOCK_PAYLOAD_BYTES {
            return Err(BlockError::PayloadOverflow { id, len: dst.len() });
        }
        self.memory.read(
            Self::block_offset(id) + BLOCK_HEADER_BYTES as u64 + offset as u64,
            dst,
        );
        Ok(())
    }

    /// Writes `src` at `offset` within block `id`'s payload (mints implicitly).
    pub fn write_payload_partial(
        &mut self,
        id: u32,
        offset: usize,
        src: &[u8],
    ) -> Result<(), BlockError> {
        if id >= self.tail_next {
            return Err(BlockError::NotMinted { id });
        }
        if offset + src.len() > BLOCK_PAYLOAD_BYTES {
            return Err(BlockError::PayloadOverflow { id, len: src.len() });
        }
        self.memory.write(
            Self::block_offset(id) + BLOCK_HEADER_BYTES as u64 + offset as u64,
            src,
        );
        Ok(())
    }

    /// Releases a block onto the free list.
    pub fn release(&mut self, id: u32) -> Result<(), BlockError> {
        if id >= self.tail_next {
            return Err(BlockError::NotMinted { id });
        }
        // Link: owner field carries next-free.
        let mut hdr = [0u8; BLOCK_HEADER_BYTES];
        hdr[0] = BlockKind::Free as u8;
        hdr[4..8].copy_from_slice(&self.free_head.to_le_bytes());
        self.memory.write(Self::block_offset(id), &hdr);
        self.free_head = id;
        self.free_count += 1;
        self.persist_header().map_err(|_| BlockError::OutOfBlocks)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::VecMemory;

    #[test]
    fn mint_write_read_roundtrip() {
        let mut store = LtbRawBlockStore::new(VecMemory::new()).unwrap();
        let a = store.mint_zeroed().unwrap();
        let b = store.mint_zeroed().unwrap();
        assert_eq!(store.minted(), 2);
        store
            .write_payload_partial(a, 0, &7u32.to_le_bytes())
            .unwrap();
        store
            .write_payload_partial(b, 4092, &99u32.to_le_bytes())
            .unwrap();
        let mut buf4 = [0u8; 4];
        store.read_payload_partial(a, 0, &mut buf4).unwrap();
        assert_eq!(u32::from_le_bytes(buf4), 7);
        store.read_payload_partial(b, 4092, &mut buf4).unwrap();
        assert_eq!(u32::from_le_bytes(buf4), 99);

        let mut full = [0u8; BLOCK_PAYLOAD_BYTES];
        store.read_payload(a, &mut full).unwrap();
        assert_eq!(u32::from_le_bytes(full[0..4].try_into().unwrap()), 7);
    }

    #[test]
    fn release_and_reuse() {
        let mut store = LtbRawBlockStore::new(VecMemory::new()).unwrap();
        let a = store.mint_zeroed().unwrap();
        let _b = store.mint_zeroed().unwrap();
        store.release(a).unwrap();
        let c = store.mint_zeroed().unwrap();
        assert_eq!(c, a); // reused the freed block
        assert_eq!(store.minted(), 2);
    }

    #[test]
    fn not_minted_access_fails() {
        let mut store = LtbRawBlockStore::new(VecMemory::new()).unwrap();
        let mut full = [0u8; BLOCK_PAYLOAD_BYTES];
        assert_eq!(
            store.read_payload(0, &mut full),
            Err(BlockError::NotMinted { id: 0 })
        );
        assert_eq!(
            store.write_payload(0, &full),
            Err(BlockError::NotMinted { id: 0 })
        );
    }
}
