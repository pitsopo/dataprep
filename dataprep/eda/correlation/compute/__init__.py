"""This module implements the intermediates computation
for plot_correlation(df) function."""

from typing import Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd

from dataprep.eda.dtypes import drop_null

from ...dtypes import NUMERICAL_DTYPES
from ...intermediate import Intermediate
from ...utils import DataArray
from .bivariate import _calc_bivariate
from .nullivariate import _calc_nullivariate
from .univariate import _calc_univariate

__all__ = ["compute_correlation"]


def compute_correlation(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Intermediate:
    """
    Parameters
    ----------
    df
        The pandas dataframe for which plots are calculated for each column.
    x
        A valid column name of the dataframe
    y
        A valid column name of the dataframe
    value_range
        If the correlation value is out of the range, don't show it.
    k
        Choose top-k element
    """

    df = df.select_dtypes(NUMERICAL_DTYPES)
    df = drop_null(df)
    # df = to_dask(df)
    df = DataArray(df)

    if x is None and y is None:  # pylint: disable=no-else-return
        return _calc_nullivariate(df, value_range=value_range, k=k)
    elif x is not None and y is None:
        return _calc_univariate(df, x=x, value_range=value_range, k=k)
    elif x is None and y is not None:
        raise ValueError("Please give the column name to x instead of y")
    elif x is not None and y is not None:
        return _calc_bivariate(df, x=x, y=y, k=k)

    raise ValueError("Not Possible")
