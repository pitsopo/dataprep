"""This module implements the intermediates computation
for plot_correlation(df) function."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import dask
import dask.array as da
import numpy as np
import pandas as pd
from bottleneck import rankdata

from ...intermediate import Intermediate
from ...utils import DataArray
from .common import CorrelationMethod, kendalltau


def _calc_nullivariate(
    df: DataArray,
    *,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Intermediate:

    if value_range is not None and k is not None:
        raise ValueError("value_range and k cannot be present in both")

    cordx, cordy, corrs = correlation_nxn(df)

    # The computations below is not expensive (scales with # of columns)
    # So we do them in pandas

    (corrs,) = dask.compute(corrs)

    columns = df.columns

    dfs = {}
    for method, corr in corrs.items():
        ndf = pd.DataFrame({"x": cordx, "y": cordy, "correlation": corr.ravel()})
        ndf = ndf[ndf["y"] > ndf["x"]]  # Retain only lower triangle (w/o diag)

        if k is not None:
            thresh = ndf["correlation"].abs().nlargest(k).iloc[-1]
            ndf = ndf[(ndf["correlation"] >= thresh) | (ndf["correlation"] <= -thresh)]
        elif value_range is not None:
            mask = (value_range[0] <= ndf["correlation"]) & (
                ndf["correlation"] <= value_range[1]
            )
            ndf = ndf[mask]

        dfs[method.name] = ndf

    return Intermediate(
        data=dfs, axis_range=list(columns.unique()), visual_type="correlation_heatmaps",
    )


def correlation_nxn(
    df: DataArray,
) -> Tuple[Sequence[str], Sequence[str], Dict[CorrelationMethod, da.Array]]:
    """
    Calculation of a n x n correlation matrix for n columns

    Returns
    -------
        The long format of the correlations
    """

    cordx, cordy = np.meshgrid(df.columns, df.columns)
    cordx, cordy = cordy.ravel(), cordx.ravel()

    corrs = {
        CorrelationMethod.Pearson: _pearson_nxn(df.values),
        CorrelationMethod.Spearman: _spearman_nxn(df.values),
        CorrelationMethod.KendallTau: _kendall_tau_nxn(df.values),
    }

    return cordx, cordy, corrs


def _pearson_nxn(data: da.Array) -> da.Array:
    """Calculate column-wise pearson correlation."""

    mean = data.mean(axis=0)[None, :]
    dem = data - mean

    num = dem.T @ dem

    std = data.std(axis=0, keepdims=True)
    dom = data.shape[0] * (std * std.T)

    correl = num / dom

    return correl


def _spearman_nxn(array: da.Array) -> da.Array:
    rank_array = (
        array.rechunk((-1, None))  #! TODO: avoid this
        .map_blocks(partial(rankdata, axis=0))
        .rechunk("auto")
    )
    return _pearson_nxn(rank_array)


def _kendall_tau_nxn(array: da.Array) -> da.Array:
    """Kendal Tau correlation outputs an n x n correlation matrix for n columns."""

    _, ncols = array.shape

    corrmat = []
    for _ in range(ncols):
        corrmat.append([float("nan")] * ncols)

    for i in range(ncols):
        corrmat[i][i] = 1.0

    for i in range(ncols):
        for j in range(i + 1, ncols):

            tmp = kendalltau(array[:, i], array[:, j])

            corrmat[j][i] = corrmat[i][j] = da.from_delayed(
                tmp, shape=(), dtype=np.float
            )

    return da.stack(corrmat)
