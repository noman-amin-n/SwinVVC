"""
Bjontegaard delta metrics (BD-BR, BD-PSNR) and encoding-time savings.

Implements the piecewise cubic-polynomial method of VCEG-M33: each RD curve is
fitted with a third-order polynomial in the log-rate domain, the two fits are
integrated over their overlapping interval, and the difference of the integrals
divided by the interval length gives the average difference.

Both directions are supported:

* ``mode="dsnr"`` fits PSNR as a function of log-rate  -> average PSNR delta in dB;
* ``mode="rate"`` fits log-rate as a function of PSNR  -> average rate delta in %.

By convention the first curve is the anchor and the second is the method under
test, so a positive BD-BR means the test method spends more bits at equal
quality, and a negative BD-PSNR means it loses quality at equal rate.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def bjontegaard(
    rate_anchor: Sequence[float],
    psnr_anchor: Sequence[float],
    rate_test: Sequence[float],
    psnr_test: Sequence[float],
    mode: str = "rate",
) -> float:
    """
    Compute a Bjontegaard delta between two RD curves.

    Args:
        rate_anchor: anchor bitrates (any consistent unit, e.g. kbps).
        psnr_anchor: anchor PSNR values in dB.
        rate_test:   test-method bitrates, same unit as `rate_anchor`.
        psnr_test:   test-method PSNR values in dB.
        mode:        ``"rate"`` for BD-BR in percent, ``"dsnr"`` for BD-PSNR
                     in dB.

    Returns:
        BD-BR (%) or BD-PSNR (dB).

    Raises:
        ValueError: on an unknown `mode`, fewer than four points per curve, or
                    non-overlapping curves.
    """
    rate_anchor = np.asarray(rate_anchor, dtype=float)
    psnr_anchor = np.asarray(psnr_anchor, dtype=float)
    rate_test = np.asarray(rate_test, dtype=float)
    psnr_test = np.asarray(psnr_test, dtype=float)

    # A cubic fit needs at least four points; the CTC uses four QPs.
    for name, array in (
        ("rate_anchor", rate_anchor),
        ("psnr_anchor", psnr_anchor),
        ("rate_test", rate_test),
        ("psnr_test", psnr_test),
    ):
        if array.size < 4:
            raise ValueError(f"{name} needs at least 4 points for a cubic fit, got {array.size}")

    log_rate_anchor = np.log(rate_anchor)
    log_rate_test = np.log(rate_test)

    if mode == "dsnr":
        # PSNR as a function of log-rate; integrate over the shared rate range.
        fit_anchor = np.polyfit(log_rate_anchor, psnr_anchor, 3)
        fit_test = np.polyfit(log_rate_test, psnr_test, 3)
        lower = max(log_rate_anchor.min(), log_rate_test.min())
        upper = min(log_rate_anchor.max(), log_rate_test.max())
    elif mode == "rate":
        # log-rate as a function of PSNR; integrate over the shared PSNR range.
        fit_anchor = np.polyfit(psnr_anchor, log_rate_anchor, 3)
        fit_test = np.polyfit(psnr_test, log_rate_test, 3)
        lower = max(psnr_anchor.min(), psnr_test.min())
        upper = min(psnr_anchor.max(), psnr_test.max())
    else:
        raise ValueError(f"Invalid mode {mode!r}; use 'dsnr' or 'rate'.")

    if upper <= lower:
        raise ValueError("The two RD curves do not overlap; cannot integrate.")

    integral_anchor = np.polyval(np.polyint(fit_anchor), upper) - np.polyval(
        np.polyint(fit_anchor), lower
    )
    integral_test = np.polyval(np.polyint(fit_test), upper) - np.polyval(
        np.polyint(fit_test), lower
    )
    average_difference = (integral_test - integral_anchor) / (upper - lower)

    if mode == "dsnr":
        return float(average_difference)
    # Undo the logarithm to express the rate difference as a percentage.
    return float((np.exp(average_difference) - 1.0) * 100.0)


def bd_rate(
    rate_anchor: Sequence[float],
    psnr_anchor: Sequence[float],
    rate_test: Sequence[float],
    psnr_test: Sequence[float],
) -> float:
    """BD-BR in percent (positive = the test method uses more bits)."""
    return bjontegaard(rate_anchor, psnr_anchor, rate_test, psnr_test, mode="rate")


def bd_psnr(
    rate_anchor: Sequence[float],
    psnr_anchor: Sequence[float],
    rate_test: Sequence[float],
    psnr_test: Sequence[float],
) -> float:
    """BD-PSNR in dB (negative = the test method loses quality)."""
    return bjontegaard(rate_anchor, psnr_anchor, rate_test, psnr_test, mode="dsnr")


def time_saving(
    time_anchor: Sequence[float],
    time_test: Sequence[float],
) -> Dict[str, object]:
    """
    Encoding-time saving per operating point and on average.

    Implements

        dT = mean over QP of (T_anchor(QP) - T_test(QP)) / T_anchor(QP)

    Args:
        time_anchor: anchor encoding times, one per QP.
        time_test:   test-method encoding times, one per QP, same order.

    Returns:
        ``{"per_qp": [...percent...], "average": percent}``.
    """
    time_anchor = np.asarray(time_anchor, dtype=float)
    time_test = np.asarray(time_test, dtype=float)

    if time_anchor.shape != time_test.shape:
        raise ValueError(
            f"Shape mismatch: anchor {time_anchor.shape} vs test {time_test.shape}"
        )

    per_qp = (time_anchor - time_test) / time_anchor * 100.0
    return {"per_qp": per_qp.tolist(), "average": float(per_qp.mean())}


def relative_difference(
    anchor: Sequence[float],
    test: Sequence[float],
) -> Dict[str, object]:
    """
    Mean relative difference, as used for dSSIM and dVMAF.

    Implements the same normalisation as the time saving, so that a positive
    value means the test method is *worse* than the anchor:

        dM = mean over QP of (M_anchor(QP) - M_test(QP)) / M_anchor(QP)

    Args:
        anchor: anchor metric values, one per QP.
        test:   test-method values, one per QP, same order.

    Returns:
        ``{"per_qp": [...percent...], "average": percent}``.
    """
    return time_saving(anchor, test)
