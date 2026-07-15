//! Attribute and dimension metadata captured at index time.
//!
//! The indexer walks the HDF5 object headers anyway; capturing attributes and
//! netCDF dimension names there makes a serialized index self-contained, so
//! consumers (e.g. the xarray backend) need no second metadata reader at open
//! time.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A single attribute value, widened to the largest of its kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttributeValue {
    Str(String),
    Strs(Vec<String>),
    Int(i64),
    Ints(Vec<i64>),
    Uint(u64),
    Uints(Vec<u64>),
    Float(f64),
    Floats(Vec<f64>),
}

/// Attributes of a group or dataset, ordered for deterministic serialization.
pub type Attributes = BTreeMap<String, AttributeValue>;

/// netCDF-internal attributes that describe file structure rather than data;
/// hidden from consumers (matching h5netcdf/netCDF4 behaviour).
const INTERNAL: &[&str] = &[
    "DIMENSION_LIST",
    "REFERENCE_LIST",
    "CLASS",
    "NAME",
    "_Netcdf4Dimid",
    "_Netcdf4Coordinates",
    "_NCProperties",
];

/// Read all (non-internal) attributes of an HDF5 group or dataset.
///
/// Unreadable attributes (exotic datatypes) are skipped rather than failing
/// the whole index.
pub fn read_attributes(loc: &hdf5::Location) -> Attributes {
    let mut attrs = Attributes::new();
    let names = match loc.attr_names() {
        Ok(names) => names,
        Err(_) => return attrs,
    };
    for name in names {
        if INTERNAL.contains(&name.as_str()) {
            continue;
        }
        match read_attribute(loc, &name) {
            Ok(value) => {
                attrs.insert(name, value);
            }
            Err(e) => {
                log::debug!("skipping unreadable attribute {name}: {e}");
            }
        }
    }
    attrs
}

fn read_attribute(loc: &hdf5::Location, name: &str) -> Result<AttributeValue, anyhow::Error> {
    use hdf5::types::TypeDescriptor as TD;
    use hdf5::types::VarLenUnicode;

    let attr = loc.attr(name)?;
    let td = attr.dtype()?.to_descriptor()?;
    // netCDF stores single-value attributes as 1-element arrays; a length-1
    // read collapses to the scalar variant either way.
    let scalar = attr.size() <= 1;

    fn collapse<T, S, V>(mut values: Vec<T>, scalar: bool, one: S, many: V) -> AttributeValue
    where
        S: FnOnce(T) -> AttributeValue,
        V: FnOnce(Vec<T>) -> AttributeValue,
    {
        if scalar && values.len() == 1 {
            one(values.remove(0))
        } else {
            many(values)
        }
    }

    Ok(match td {
        TD::Integer(_) | TD::Boolean | TD::Enum(_) => collapse(
            attr.read_raw::<i64>()?,
            scalar,
            AttributeValue::Int,
            AttributeValue::Ints,
        ),
        TD::Unsigned(_) => collapse(
            attr.read_raw::<u64>()?,
            scalar,
            AttributeValue::Uint,
            AttributeValue::Uints,
        ),
        TD::Float(_) => collapse(
            attr.read_raw::<f64>()?,
            scalar,
            AttributeValue::Float,
            AttributeValue::Floats,
        ),
        // libhdf5 registers no fixed<->vlen string conversion (h5py performs
        // that conversion in software), so fixed strings are read raw with the
        // file datatype and trimmed here.
        TD::FixedAscii(n) | TD::FixedUnicode(n) => collapse(
            read_fixed_strings(&attr, n)?,
            scalar,
            AttributeValue::Str,
            AttributeValue::Strs,
        ),
        TD::VarLenAscii => {
            use hdf5::types::VarLenAscii;

            collapse(
                attr.read_raw::<VarLenAscii>()?
                    .into_iter()
                    .map(|s| s.to_string())
                    .collect(),
                scalar,
                AttributeValue::Str,
                AttributeValue::Strs,
            )
        }
        TD::VarLenUnicode => collapse(
            attr.read_raw::<VarLenUnicode>()?
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            scalar,
            AttributeValue::Str,
            AttributeValue::Strs,
        ),
        other => anyhow::bail!("unsupported attribute type: {other:?}"),
    })
}

/// Resolve the netCDF dimension names of a dataset, in order.
///
/// netCDF-4 stores each variable's dimensions as the ``DIMENSION_LIST``
/// attribute: one variable-length array of object references per dimension,
/// pointing at the dimension-scale dataset. A dimension scale's own dimension
/// is itself (by netCDF convention its name); a plain dataset without
/// ``DIMENSION_LIST`` (e.g. HDF-EOS5 or plain HDF5) falls back to
/// ``phony_dim_<i>`` names, matching h5netcdf.
pub fn dimension_names(ds: &hdf5::Dataset) -> Vec<String> {
    use hdf5::types::VarLenArray;
    use hdf5::ObjectReference1;

    let ndim = ds.ndim();

    if is_dimension_scale(ds) {
        return vec![basename(&ds.name())];
    }

    if let Ok(attr) = ds.attr("DIMENSION_LIST") {
        if let Ok(refs) = attr.read_raw::<VarLenArray<ObjectReference1>>() {
            let names: Vec<Option<String>> = refs
                .iter()
                .map(|vl| {
                    vl.iter().next().and_then(|r| match ds.dereference(r) {
                        Ok(hdf5::ReferencedObject::Dataset(dim)) => Some(basename(&dim.name())),
                        _ => None,
                    })
                })
                .collect();
            if names.len() == ndim && names.iter().all(Option::is_some) {
                return names.into_iter().flatten().collect();
            }
        }
    }

    (0..ndim).map(|i| format!("phony_dim_{i}")).collect()
}

/// Read a fixed-length string attribute raw, using the FILE datatype (no
/// conversion involved), and trim trailing NULs/padding per element.
fn read_fixed_strings(attr: &hdf5::Attribute, len: usize) -> Result<Vec<String>, anyhow::Error> {
    use hdf5::h5check;

    let n = attr.size().max(1);
    let mut buf = vec![0u8; n * len];
    {
        // Raw FFI must hold the library lock the crate's own calls take:
        // libhdf5 is not thread-safe, and an unsynchronized H5Aread aborts in
        // H5SL under concurrent access.
        let _guard = hdf5_sys::LOCK.lock();
        unsafe {
            let file_type = h5check(hdf5_sys::h5a::H5Aget_type(attr.id()))?;
            let res = hdf5_sys::h5a::H5Aread(attr.id(), file_type, buf.as_mut_ptr().cast());
            hdf5_sys::h5t::H5Tclose(file_type);
            anyhow::ensure!(res >= 0, "H5Aread failed for fixed string attribute");
        }
    }
    Ok(buf
        .chunks_exact(len)
        .map(|chunk| {
            let end = chunk.iter().position(|&b| b == 0).unwrap_or(len);
            String::from_utf8_lossy(&chunk[..end]).trim_end().to_string()
        })
        .collect())
}

fn is_dimension_scale(ds: &hdf5::Dataset) -> bool {
    matches!(
        read_attribute(ds, "CLASS"),
        Ok(AttributeValue::Str(class)) if class == "DIMENSION_SCALE"
    )
}

fn basename(path: &str) -> String {
    path.rsplit('/').next().unwrap_or(path).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::idx::Index;

    #[test]
    fn coads_attributes_and_dims() {
        let hf = hdf5::File::open("tests/data/coads_climatology.nc4").unwrap();

        let root = read_attributes(&hf);
        println!("global attrs: {root:#?}");

        let sst = hf.dataset("SST").unwrap();
        let attrs = read_attributes(&sst);
        assert_eq!(attrs.get("units"), Some(&AttributeValue::Str("Deg C".into())));
        assert!(attrs.contains_key("long_name"));
        assert!(attrs.contains_key("missing_value"));
        assert!(!attrs.contains_key("DIMENSION_LIST"));

        let dims = dimension_names(&sst);
        assert_eq!(dims, vec!["TIME", "COADSY", "COADSX"]);

        // a dimension scale names itself
        let time = hf.dataset(&dims[0]).unwrap();
        assert_eq!(dimension_names(&time), vec![dims[0].clone()]);

        let _ = Index::index("tests/data/coads_climatology.nc4").unwrap();
    }
}

