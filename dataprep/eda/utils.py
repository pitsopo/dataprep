"""Miscellaneous functions
"""
import logging
from math import ceil
from typing import Any, Tuple, Union, cast

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from bokeh.models import Legend
from bokeh.plotting import Figure

LOGGER = logging.getLogger(__name__)


def is_notebook() -> Any:
    """
    :return: whether it is running in jupyter notebook
    """
    try:
        # pytype: disable=import-error
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel

        # pytype: enable=import-error

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        return False
    except (NameError, ImportError):
        return False


def to_dask(df: Union[pd.DataFrame, dd.DataFrame]) -> dd.DataFrame:
    """
    Convert a dataframe to a dask dataframe.
    """
    if isinstance(df, dd.DataFrame):
        return df

    df_size = df.memory_usage(deep=True).sum()
    npartitions = ceil(df_size / 128 / 1024 / 1024)  # 128 MB partition size
    return dd.from_pandas(df, npartitions=npartitions)


def sample_n(arr: np.ndarray, n: int) -> np.ndarray:  # pylint: disable=C0103
    """
    Sample n values uniformly from the range of the `arr`,
    not from the distribution of `arr`'s elems.
    """
    if len(arr) <= n:
        return arr

    subsel = np.linspace(0, len(arr) - 1, n)
    subsel = np.floor(subsel).astype(int)
    return arr[subsel]


def relocate_legend(fig: Figure, loc: str) -> Figure:
    """
    Relocate legend(s) from center to `loc`
    """
    remains = []
    targets = []
    for layout in fig.center:
        if isinstance(layout, Legend):
            targets.append(layout)
        else:
            remains.append(layout)
    fig.center = remains
    for layout in targets:
        fig.add_layout(layout, loc)

    return fig


def cut_long_name(name: str, max_len: int = 12) -> str:
    """
    If the name is longer than `max_len`,
    cut it to `max_len` length and append "..."
    """
    # Bug 136 Fixed
    name = str(name)
    if len(name) <= max_len:
        return name
    return f"{name[:max_len]}..."


def fuse_missing_perc(name: str, perc: float) -> str:
    """
    Append (x.y%) to the name if `perc` is not 0
    """
    if perc == 0:
        return name

    return f"{name} ({perc:.1%})"


class DataArray:
    """DataArray is similar to dd.DataFrame, but already
    has its data stored in an da.Array with known chunk size.

    Creating a DataArray requires a small read on the data length.
    """

    _ddf: dd.DataFrame
    _data: da.Array
    _columns: pd.Index

    def __init__(self, df: Union[pd.DataFrame, dd.DataFrame]) -> None:
        sup = super()

        if isinstance(df, dd.DataFrame):
            sup.__setattr__("_ddf", df)
        else:
            df_size = df.memory_usage(deep=True).sum()
            npartitions = ceil(df_size / 128 / 1024 / 1024)
            sup.__setattr__("_ddf", dd.from_pandas(df, npartitions=npartitions))

        sup.__setattr__("_data", self._ddf.to_dask_array(lengths=True))
        sup.__setattr__("_columns", pd.Index([str(col) for col in self._ddf.columns]))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the data"""
        return cast(Tuple[int, ...], self._data.shape)

    @property
    def values(self) -> da.Array:
        """Return the array representation of the data"""
        return self._data

    @property
    def columns(self) -> pd.Index:
        """Return the columns of the DataFrame"""
        return self._columns
