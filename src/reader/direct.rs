use crate::filters::byteorder::Order;
use std::fs::File;
use std::path::{Path, PathBuf};

use super::{
    dataset::{ParReader, Reader},
    par::ParSourceReader,
};
use crate::extent::Extents;
use crate::idx::Dataset;

/// A parallel reader opening the file at `path` once per worker: the
/// [`File::open`] specialization of [`ParSourceReader`].
pub struct Direct<'a, const D: usize> {
    ds: &'a Dataset<'a, D>,
    path: PathBuf,
}

impl<'a, const D: usize> Direct<'a, D> {
    pub fn with_dataset<P: AsRef<Path>>(
        ds: &'a Dataset<D>,
        path: P,
    ) -> Result<Direct<'a, D>, anyhow::Error> {
        Ok(Direct {
            ds,
            path: path.as_ref().into(),
        })
    }

    fn source(
        &self,
    ) -> Result<ParSourceReader<'_, impl Fn() -> std::io::Result<File> + Sync + '_, D>, anyhow::Error>
    {
        ParSourceReader::with_dataset(self.ds, || File::open(&self.path))
    }
}

impl<const D: usize> ParReader for Direct<'_, D> {
    fn read_to_par(&self, extents: &Extents, dst: &mut [u8]) -> Result<usize, anyhow::Error> {
        self.source()?.read_to_par(extents, dst)
    }
}

impl<const D: usize> Reader for Direct<'_, D> {
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
        self.source()?.read_to(extents, dst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idx::DatasetD;
    use crate::prelude::*;

    #[test]
    fn read_coads_sst() {
        let i = Index::index("tests/data/coads_climatology.nc4").unwrap();
        let ds = if let DatasetD::D3(ds) = i.dataset("SST").unwrap() {
            ds
        } else {
            panic!()
        };
        let mut r = Direct::with_dataset(ds, i.path().unwrap()).unwrap();

        let vs = r.values::<f32, _>(..).unwrap();
        let h = hdf5::File::open(i.path().unwrap()).unwrap();
        let hvs = h.dataset("SST").unwrap().read_raw::<f32>().unwrap();

        assert_eq!(vs, hvs);
    }
}
