"""This module implements the intermediates computation
for plot_correlation(df) function."""

from operator import itruediv
from typing import Optional, Tuple

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np

from ...utils import DataArray
from ...intermediate import Intermediate


def _calc_bivariate(
    df: DataArray,
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    k: Optional[int] = None,
) -> Intermediate:
    if x not in df.columns:
        raise ValueError(f"{x} not in columns names")
    if y not in df.columns:
        raise ValueError(f"{y} not in columns names")

    xloc = df.columns.get_loc(x)
    yloc = df.columns.get_loc(y)

    coeffs, ndf, influences = scatter_with_regression(
        df.values[:, xloc], df.values[:, yloc], k=k, sample_size=1000,
    )
    coeffs, ndf, influences = dask.compute(coeffs, ndf, influences)

    # lazy/eager border line
    result = {
        "coeffs": coeffs,
        "data": ndf.rename(columns={"x": x, "y": y}),
    }

    if (influences is None) != (k is None):
        raise RuntimeError("Not possible")

    if influences is not None and k is not None:
        infidx = np.argsort(influences)
        labels = np.full(len(influences), "=")
        # pylint: disable=invalid-unary-operand-type
        labels[infidx[-k:]] = "-"  # type: ignore
        # pylint: enable=invalid-unary-operand-type
        labels[infidx[:k]] = "+"
        result["data"]["influence"] = labels

    return Intermediate(**result, visual_type="correlation_scatter")


def scatter_with_regression(
    xarr: da.Array, yarr: da.Array, sample_size: int, k: Optional[int] = None
) -> Tuple[Tuple[da.Array, da.Array], dd.DataFrame, Optional[da.Array]]:
    """Calculate pearson correlation on 2 given arrays.

    Parameters
    ----------
    xarr : da.Array
    yarr : da.Array
    sample_size : int
    k : Optional[int] = None
        Highlight k points which influence pearson correlation most

    Returns
    -------
    Intermediate
    """
    if k == 0:
        raise ValueError("k should be larger than 0")

    xarrp1 = da.vstack([xarr, da.ones_like(xarr)]).T
    xarrp1 = xarrp1.rechunk((xarrp1.chunks[0], -1))
    (coeffa, coeffb), _, _, _ = da.linalg.lstsq(xarrp1, yarr)

    if sample_size < xarr.shape[0]:
        samplesel = np.random.choice(xarr.shape[0], int(sample_size))
        xarr = xarr[samplesel]
        yarr = yarr[samplesel]

    df = dd.concat([dd.from_dask_array(arr) for arr in [xarr, yarr]], axis=1)
    df.columns = ["x", "y"]

    if k is None:
        return (coeffa, coeffb), df, None

    influences = pearson_influence(xarr, yarr)
    return (coeffa, coeffb), df, influences


def pearson_influence(xarr: da.Array, yarr: da.Array) -> da.Array:
    """Calculating the influence for deleting a point on the pearson correlation"""

    if xarr.shape != yarr.shape:
        raise ValueError(
            f"The shape of xarr and yarr should be same, got {xarr.shape}, {yarr.shape}"
        )

    # Fast calculating the influence for removing one element on the correlation
    n = xarr.shape[0]

    x2, y2 = da.square(xarr), da.square(yarr)
    xy = xarr * yarr

    # The influence is vectorized on xarr and yarr, so we need to repeat all the sums for n times

    xsum = da.ones(n) * da.sum(xarr)
    ysum = da.ones(n) * da.sum(yarr)
    xysum = da.ones(n) * da.sum(xy)
    x2sum = da.ones(n) * da.sum(x2)
    y2sum = da.ones(n) * da.sum(y2)

    # Note: in we multiply (n-1)^2 to both denominator and numerator to avoid divisions.
    numerator = (n - 1) * (xysum - xy) - (xsum - xarr) * (ysum - yarr)

    varx = (n - 1) * (x2sum - x2) - da.square(xsum - xarr)
    vary = (n - 1) * (y2sum - y2) - da.square(ysum - yarr)
    denominator = da.sqrt(varx * vary)

    return da.map_blocks(itruediv, numerator, denominator, dtype=numerator.dtype)
