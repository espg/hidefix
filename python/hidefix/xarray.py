"""
An Xarray backend based on hidefix.
"""

import os
import logging

import hidefix
import numpy as np

from xarray.backends.common import (
    BackendArray,
    BackendEntrypoint,
    WritableCFDataStore,
    _normalize_path,
)

from xarray.core import indexing
from xarray.backends.store import StoreBackendEntrypoint
from xarray.coding.variables import pop_to
from xarray.core.variable import Variable

from xarray.core.utils import FrozenDict

logger = logging.getLogger(__name__)


class HidefixBackendEntrypoint(BackendEntrypoint):

    available = True
    description = "Open netCDF4 files with the multi-threaded Hidefix backend in Xarray"
    url = "https://github.com/gauteh/hidefix"
    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        use_cftime=None,
        decode_timedelta=None,
        group=None,
    ):
        # TODO: allow take an existing index, maybe from a serialized object.

        filename_or_obj = _normalize_path(filename_or_obj)

        store = HidefixDataStore.open(filename_or_obj, group)

        store_entrypoint = StoreBackendEntrypoint()
        return store_entrypoint.open_dataset(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".nc", ".nc4", ".cdf"}


class HidefixDataStore(WritableCFDataStore):
    idx: hidefix.Index
    path: str
    group: str

    def __init__(self, path, group=None):
        self.path = path
        self.group = group

        self.idx = hidefix.Index(path)

    @classmethod
    def open(
        cls,
        filename,
        group,
    ):
        if isinstance(filename, os.PathLike):
            filename = os.fspath(filename)

        if not isinstance(filename, str):
            raise ValueError(
                "the hidefix backend can only read file-like objects")

        return cls(filename, group)

    def get_attrs(self):
        return FrozenDict(self.idx.attributes(self.group))

    def get_dimensions(self):
        # dimension sizes from the per-variable dims/shape zip; a dimension
        # scale names itself, so coordinate variables are covered too.
        dims = {}
        for name in self.idx.datasets(self.group):
            ds = self.idx.dataset(name, self.group)
            for dim, size in zip(self.idx.dataset_dims(name, self.group),
                                 ds.shape()):
                dims[dim] = int(size)
        return FrozenDict(dims)

    def get_encoding(self):
        # unlimited dimensions are not captured by the index.
        return {"unlimited_dims": set()}

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k)) for k in self.idx.datasets(self.group))

    def open_store_variable(self, k):
        ds = self.idx.dataset(k, self.group)
        attributes = self.idx.dataset_attributes(k, self.group)
        dimensions = tuple(self.idx.dataset_dims(k, self.group))

        data = indexing.LazilyIndexedArray(
            HidefixArray(self, k, self.group, ds, attributes))

        # fill values are applied (as NaN) by HidefixArray.
        attributes.pop('_FillValue', None)
        attributes.pop('missing_value', None)

        encoding = {}
        pop_to(attributes, encoding, "least_significant_digit")
        # save source so __repr__ can detect if it's local or not
        encoding["source"] = self.path
        encoding["original_shape"] = data.shape
        encoding["dtype"] = data.dtype

        return Variable(dimensions, data, attributes, encoding)


class HidefixArray(BackendArray):

    def __init__(self, store, name, group, ds, attributes):
        self.store = store
        self.variable_name = name
        self.group = group

        self.shape = tuple(int(s) for s in ds.shape())
        self.dtype = np.dtype(ds.dtype())
        self.fill_value = attributes.get('_FillValue', None)
        missing = attributes.get('missing_value', None)
        if missing is not None:
            if self.fill_value is None:
                self.fill_value = missing
            else:
                assert missing == self.fill_value, "mismatch between missing_value and _FillValue"

    def __getitem__(self, key):
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem)

    def _getitem(self, key):
        #TODO: perf: cache this? maybe this is making single value access slow.
        array = self.store.idx.dataset(self.variable_name, self.group)
        data = array[key]
        if self.fill_value is not None:
            array.apply_fill_value(self.fill_value, np.nan, data)
        return data
