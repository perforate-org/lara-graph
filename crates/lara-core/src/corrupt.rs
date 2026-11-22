//! The response to inconsistent persistent metadata.

/// Stored metadata is inconsistent. Not recoverable: stop rather than corrupt further. On the
/// IC the trap rolls the whole message back.
#[cold]
pub(crate) fn corrupt(what: &str) -> ! {
    panic!("graph storage is corrupt: {what}")
}

/// Memory reserved before a multi-part update ran out: the reservation undercounted. A bug, not
/// a storage fault; the trap keeps the update from landing half-done.
#[cold]
pub(crate) fn reservation_exceeded() -> ! {
    panic!("an update needed more memory than it reserved")
}
