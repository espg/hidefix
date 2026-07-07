//! Parallel reader for sources that can hand out an independent [`Read`] + [`Seek`]
//! handle per rayon job (never shared across threads).
//!
//! [`ParSourceReader`] generalizes [`Direct`](super::direct::Direct) (which opens the
//! file once per rayon worker) to any handle factory. This makes parallel decoding
//! available to in-memory or network-fed sources, e.g. pre-fetched chunk buffers
//! wrapped in a [`Cursor`](std::io::Cursor):
//!
//! ```
//! use std::io::Cursor;
//! use hidefix::prelude::*;
//! use hidefix::idx::DatasetD;
//! use hidefix::reader::par::ParSourceReader;
//!
//! let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
//! let DatasetD::D3(ds) = i.dataset("SST").unwrap() else { panic!() };
//!
//! let bytes = std::fs::read("tests/data/coads_climatology.nc4").unwrap();
//! let r = ParSourceReader::with_dataset(ds, || Ok(Cursor::new(&bytes[..]))).unwrap();
//!
//! let values = r.values_par::<f32, _>(..).unwrap();
//! ```
use anyhow::ensure;
use std::io::{Read, Seek};

use super::{
    chunk::{decode_chunk, read_chunk, read_chunk_to},
    dataset::{ParReader, Reader},
};
use crate::extent::Extents;
use crate::filters::byteorder::Order;
use crate::idx::{Chunk, Dataset};

/// A reader decoding chunks in parallel (with `rayon`), where each worker reads from an
/// independent [`Read`] + [`Seek`] handle produced by the `source` factory.
pub struct ParSourceReader<'a, F, const D: usize> {
    ds: &'a Dataset<'a, D>,
    source: F,
    chunk_sz: u64,
}

impl<'a, F, R, const D: usize> ParSourceReader<'a, F, D>
where
    F: Fn() -> std::io::Result<R>,
    R: Read + Seek,
{
    /// Creates a reader for `ds`, where `source` produces one independent handle per
    /// worker (e.g. `|| File::open(path)`, or a [`Cursor`](std::io::Cursor) over an
    /// in-memory buffer).
    pub fn with_dataset(
        ds: &'a Dataset<D>,
        source: F,
    ) -> Result<ParSourceReader<'a, F, D>, anyhow::Error> {
        let chunk_sz = ds.chunk_shape.iter().product::<u64>() * ds.dsize as u64;

        Ok(ParSourceReader {
            ds,
            source,
            chunk_sz,
        })
    }
}

impl<F, R, const D: usize> ParReader for ParSourceReader<'_, F, D>
where
    F: Fn() -> std::io::Result<R> + Sync,
    R: Read + Seek,
{
    fn read_to_par(&self, extents: &Extents, dst: &mut [u8]) -> Result<usize, anyhow::Error> {
        use rayon::prelude::*;

        let counts = extents.get_counts(self.shape())?;

        let dsz = self.ds.dsize as u64;
        let vsz = counts.product::<u64>() * dsz;

        ensure!(
            dst.len() >= vsz as usize,
            "destination buffer has insufficient capacity"
        );

        let groups = self.ds.group_chunk_slices(extents);
        let groups = groups.chunk_by(|a, b| a.0.addr == b.0.addr);
        let groups = groups.collect::<Vec<_>>();

        groups.par_iter().try_for_each_init(
            || (self.source)(),
            |fd, group| {
                let fd = fd
                    .as_mut()
                    .map_err(|e| anyhow::anyhow!("could not open source: {e}"))?;
                let c = group[0].0;

                let mut chunk: Vec<u8> = vec![0; c.size.get() as usize];
                read_chunk_to(fd, c.addr.get(), &mut chunk)?;

                let chunk = decode_chunk(
                    chunk,
                    self.chunk_sz,
                    dsz,
                    self.ds.gzip.is_some(),
                    self.ds.shuffle,
                )?;

                for (_c, current, start, end) in *group {
                    let start = (start * dsz) as usize;
                    let end = (end * dsz) as usize;
                    let current = (current * dsz) as usize;

                    debug_assert!(start <= chunk.len());
                    debug_assert!(end <= chunk.len());

                    let sz = end - start;

                    // Safety: The sub-slices never overlap between threads and segments. But I
                    // cannot find a good way to do this in Rust at the moment. Maybe with a
                    // slice::split_at_indices method or equivalent that gives a new slice of sub-slices.
                    let dptr = dst[current..].as_ptr() as _;
                    let src = chunk[start..end].as_ptr();

                    unsafe {
                        core::ptr::copy_nonoverlapping(src, dptr, sz);
                    }
                }

                Ok::<_, anyhow::Error>(())
            },
        )?;

        Ok(vsz as usize)
    }
}

impl<F, R, const D: usize> Reader for ParSourceReader<'_, F, D>
where
    F: Fn() -> std::io::Result<R>,
    R: Read + Seek,
{
    fn order(&self) -> Order {
        self.ds.order
    }

    fn dsize(&self) -> usize {
        self.ds.dsize
    }

    fn shape(&self) -> &[u64] {
        &self.ds.shape
    }

    fn read_to(&mut self, extents: &Extents, dst: &mut [u8]) -> Result<usize, anyhow::Error> {
        let counts = extents.get_counts(self.shape())?;

        let dsz = self.ds.dsize as u64;
        let vsz = counts.product::<u64>() * dsz;

        ensure!(
            dst.len() >= vsz as usize,
            "destination buffer has insufficient capacity"
        );

        let groups = self.ds.group_chunk_slices(extents);

        let mut fd = (self.source)()?;

        let mut last_chunk: Option<(&Chunk<D>, Vec<u8>)> = None;

        for (c, current, start, end) in groups {
            let cache = match (last_chunk.as_mut(), c) {
                (Some((last, cache)), c) if c.addr == last.addr => {
                    cache // still on same
                }
                _ => {
                    // Read new chunk
                    let cache = read_chunk(
                        &mut fd,
                        c.addr.get(),
                        c.size.get(),
                        self.chunk_sz,
                        dsz,
                        self.ds.gzip.is_some(),
                        self.ds.shuffle,
                        false,
                    )?;

                    last_chunk = Some((c, cache));
                    &last_chunk.as_mut().unwrap().1
                }
            };

            let start = (start * dsz) as usize;
            let end = (end * dsz) as usize;
            let current = (current * dsz) as usize;

            debug_assert!(start <= cache.len());
            debug_assert!(end <= cache.len());

            let sz = end - start;

            // TODO: Make sure `dst` and `cache` are aligned: copying could be SIMD-ifyed.

            dst[current..(current + sz)].copy_from_slice(&cache[start..end]);
        }

        Ok(vsz as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idx::DatasetD;
    use crate::prelude::*;
    use crate::reader::cache::CacheReader;
    use std::fs::File;
    use std::io::Cursor;

    #[test]
    fn read_coads_sst() {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let DatasetD::D3(ds) = i.dataset("SST").unwrap() else {
            panic!()
        };
        let r =
            ParSourceReader::with_dataset(ds, || File::open("tests/data/coads_climatology.nc4"))
                .unwrap();

        let vs = r.values_par::<f32, _>(..).unwrap();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("SST").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_coads_sst_in_memory() {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let DatasetD::D3(ds) = i.dataset("SST").unwrap() else {
            panic!()
        };
        let bytes = std::fs::read("tests/data/coads_climatology.nc4").unwrap();
        let r = ParSourceReader::with_dataset(ds, || Ok(Cursor::new(&bytes[..]))).unwrap();

        let vs = r.values_par::<f32, _>(..).unwrap();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("SST").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_shufzip_2d_in_memory() {
        let i = Index::index("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
        let DatasetD::D2(ds) = i.dataset("d_4_shufzip_chunks").unwrap() else {
            panic!()
        };
        let bytes = std::fs::read("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
        let r = ParSourceReader::with_dataset(ds, || Ok(Cursor::new(&bytes[..]))).unwrap();

        let vs = r.values_par::<f32, _>(..).unwrap();

        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h
            .dataset("d_4_shufzip_chunks")
            .unwrap()
            .read_raw::<f32>()
            .unwrap();

        assert_eq!(vs, hvs);
    }

    #[test]
    fn read_chunked_shufzip_2d_in_memory_subset() {
        let i = Index::index("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();
        let DatasetD::D2(ds) = i.dataset("d_4_shufzip_chunks").unwrap() else {
            panic!()
        };
        let bytes = std::fs::read("tests/data/dmrpp/chunked_shufzip_twoD.h5").unwrap();

        // Chunks are 50 x 50 in a 100 x 100 dataset: the extents cut partially into all
        // four chunks.
        let extents = [25..75, 30..80];

        let r = ParSourceReader::with_dataset(ds, || Ok(Cursor::new(&bytes[..]))).unwrap();
        let vs = r.values_par::<f32, _>(extents.clone()).unwrap();

        let mut c = CacheReader::with_dataset(ds, Cursor::new(&bytes[..])).unwrap();
        let cvs = c.values::<f32, _>(extents).unwrap();

        assert_eq!(vs, cvs);
    }
}
