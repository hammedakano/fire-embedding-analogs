"""Paired statistical comparisons between retrieval methods.

Every method predicts the same withheld fires, so the comparisons are paired.
Absolute errors are compared with the Wilcoxon signed-rank test and
correlations with Steiger's test for dependent correlations, which is the
appropriate test when both correlations share the observed variable. All
p-values in a family are adjusted together with the Holm procedure.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def steiger_test(r12: float, r13: float, r23: float, n: int) -> tuple:
    """Steiger's z-test for two dependent correlations sharing one variable.

    Parameters
    ----------
    r12
        Correlation between the observed values and the first prediction.
    r13
        Correlation between the observed values and the second prediction.
    r23
        Correlation between the two predictions, that is, the information they
        share.
    n
        Number of paired observations.

    Returns
    -------
    tuple
        ``(z, p)`` for the two-sided test that ``r12`` differs from ``r13``.
    """
    r_bar = (r12 + r13) / 2.0
    determinant = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23

    z12 = np.arctanh(r12)
    z13 = np.arctanh(r13)

    covariance = (r23 * (1 - r_bar**2) ** 2) / (
        2 * (1 - r_bar**2) ** 2 + r_bar**2 * determinant
    )
    standard_error = np.sqrt((1 / (n - 3)) * (1 / (1 - covariance)))
    z = (z12 - z13) / standard_error
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def compare_methods(
    loo: pd.DataFrame,
    reference: str = "pred_emb",
    comparisons: Optional[Dict[str, str]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare the reference method against each baseline.

    Wilcoxon tests the paired absolute errors. Steiger tests the difference in
    correlation with the observed high-severity fraction. Both families of
    p-values are pooled and Holm-adjusted together, matching the published
    analysis.
    """
    comparisons = comparisons or {
        "Geographic": "pred_geo",
        "Seasonal": "pred_sea",
        "Geo + Seasonal": "pred_gs",
    }

    observed = loo["obs_high_frac"]
    reference_pred = loo[reference]
    reference_error = (reference_pred - observed).abs()
    n = len(loo)

    r_reference = float(stats.pearsonr(observed, reference_pred)[0])

    rows, steiger_p, wilcoxon_p = [], [], []
    for label, column in comparisons.items():
        baseline_pred = loo[column]
        baseline_error = (baseline_pred - observed).abs()

        r_baseline = float(stats.pearsonr(observed, baseline_pred)[0])
        r_shared = float(stats.pearsonr(reference_pred, baseline_pred)[0])
        z, p_steiger = steiger_test(r_reference, r_baseline, r_shared, n)
        w_stat, p_wilcoxon = stats.wilcoxon(reference_error, baseline_error)

        steiger_p.append(p_steiger)
        wilcoxon_p.append(p_wilcoxon)
        rows.append(
            {
                "Comparison": f"Embedding vs {label}",
                "r embedding": round(r_reference, 4),
                "r baseline": round(r_baseline, 4),
                "r shared": round(r_shared, 4),
                "Steiger z": round(z, 4),
                "Steiger p": p_steiger,
                "Wilcoxon W": round(float(w_stat), 0),
                "Wilcoxon p": p_wilcoxon,
            }
        )

    all_p = steiger_p + wilcoxon_p
    if all_p:
        _, adjusted, _, _ = multipletests(all_p, alpha=alpha, method="holm")
        for i, row in enumerate(rows):
            row["Steiger Holm-adj p"] = round(float(adjusted[i]), 4)
            row["Wilcoxon Holm-adj p"] = round(float(adjusted[i + len(rows)]), 4)

    return pd.DataFrame(rows)
