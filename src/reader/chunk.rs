//! Building blocks for reading and decoding chunks.
//!
//! These are the primitives the readers in this module are built from. Together with
//! [`Dataset::group_chunk_slices`](crate::idx::Dataset::group_chunk_slices) they can be
//! used to drive a custom decode loop when none of the provided readers fit the source.
use crate::filters;
use std::io::{Read, Seek, SeekFrom};

/// Reads the chunk (as stored in the file) at byte address `addr` in `fd` into `dst`,
/// which must have the stored (compressed) size of the chunk.
///
/// ```
/// use std::io::Cursor;
/// use hidefix::reader::chunk::read_chunk_to;
///
/// let mut fd = Cursor::new(vec![0u8, 1, 2, 3, 4, 5]);
/// let mut dst = vec![0u8; 2];
/// read_chunk_to(&mut fd, 2, &mut dst).unwrap();
/// assert_eq!(dst, [2, 3]);
/// ```
pub fn read_chunk_to<F>(fd: &mut F, addr: u64, dst: &mut [u8]) -> Result<(), anyhow::Error>
where
    F: Read + Seek,
{
    fd.seek(SeekFrom::Start(addr))?;
    fd.read_exact(dst)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn read_chunk<F>(
    fd: &mut F,
    addr: u64,
    size: u64,
    chunk_sz: u64,
    dsz: u64,
    gzipped: bool,
    shuffled: bool,
    tokio_block: bool,
) -> Result<Vec<u8>, anyhow::Error>
where
    F: Read + Seek,
{
    let mut cache: Vec<u8> = vec![0; size as usize];

    read_chunk_to(fd, addr, &mut cache)?;

    if tokio_block {
        tokio::task::block_in_place(|| decode_chunk(cache, chunk_sz, dsz, gzipped, shuffled))
    } else {
        decode_chunk(cache, chunk_sz, dsz, gzipped, shuffled)
    }
}

/// Decodes a chunk (as stored in the file) into raw values: decompresses (gzip) and
/// unshuffles according to the dataset's filters. `chunk_sz` is the decoded chunk size
/// in bytes ([`Dataset::chunk_shape`](crate::idx::Dataset::chunk_shape) product times
/// `dsz`) and `dsz` the size of the datatype in bytes.
///
/// ```
/// use hidefix::reader::chunk::decode_chunk;
///
/// // An unfiltered chunk passes through unchanged.
/// let chunk = vec![1u8, 2, 3, 4];
/// let values = decode_chunk(chunk.clone(), 4, 4, false, false).unwrap();
/// assert_eq!(values, chunk);
/// ```
pub fn decode_chunk(
    chunk: Vec<u8>,
    chunk_sz: u64,
    dsz: u64,
    gzipped: bool,
    shuffled: bool,
) -> Result<Vec<u8>, anyhow::Error> {
    debug_assert!(dsz < 16); // unlikely data-size

    // Decompress
    let cache = if gzipped {
        let mut decache = vec![0; chunk_sz as usize];

        filters::gzip::decompress(&chunk, &mut decache)?;

        debug_assert_eq!(decache.len(), chunk_sz as usize);

        decache
    } else {
        chunk
    };

    // Unshuffle
    // TODO: Keep buffers around to avoid allocations.
    // TODO: Write directly to buf_slice when on last filter.
    let cache = if shuffled && dsz > 1 {
        filters::shuffle::unshuffle_sized(&cache, dsz as usize)
    } else {
        cache
    };

    // TODO:
    // * more filters..

    Ok(cache)
}
