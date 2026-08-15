"""Turn the GP's distress probability into the published 0-100 score.

Three steps, each measured rather than assumed (see index/EXPERIMENTS.md):

1. **Calibrate.** The raw GPC probability overstates risk about 3x (predicted
   mean 0.112 vs actual 0.036). Platt scaling fitted on a held-out calibration
   year cuts Brier 0.0416 -> 0.0342 while leaving the ranking untouched
   (lift stays 2.7x). Isotonic was rejected: it flattened the ranking
   (lift 2.7x -> 1.9x) and pushed 98% of banks into one band.

2. **Map to 0-100 by anchoring on the calibration set's quantiles**, NOT by
   `(1 - prob) * 100`. Calibration pulls probabilities down towards the 3.6%
   base rate, so the naive map compresses every bank into [92, 96] and 100% of
   them land in "sound" — the mentor's <=80 cutoff would need a >20% predicted
   probability, which a calibrated model of a 3.6%-base-rate event never emits.
   Anchoring instead maps the calibration set's 70th/90th percentile onto the
   90/80 cutoffs, so the published bands keep their intended meaning.

3. **Band** on the mentor-agreed cutoffs (>=90 sound / 80-90 neutral /
   <=80 distress). Verified monotone on the test split: distress 7.7% actual
   event rate, neutral 4.1%, sound 1.6% — a 5x spread, with 19/31/50% of banks
   in each band.

Confidence intervals come from the GP latent variance. Score is a monotone
transform of the latent, so latent quantiles map onto score quantiles with the
endpoints flipped (a HIGH latent = high distress risk = LOW score).
"""

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

BAND_SOUND_MIN = 90.0      # >= 90 -> sound
BAND_NEUTRAL_MIN = 80.0    # 80-90 -> neutral, <= 80 -> distress
Z80, Z95 = norm.ppf(0.90), norm.ppf(0.975)


def fit_calibrator(prob_cal: np.ndarray, y_cal: np.ndarray) -> LogisticRegression:
    """Platt scaling on a held-out year — never on the test split."""
    return LogisticRegression().fit(prob_cal.reshape(-1, 1), y_cal)


def calibrate(calibrator: LogisticRegression, prob: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]


def fit_score_anchors(prob_cal: np.ndarray) -> dict:
    """Anchor points mapping calibrated probability onto the 0-100 score.

    The calibration set's 70th/90th percentile become the 90/80 band cutoffs,
    so band membership is a statement about where a bank sits relative to the
    population the model was calibrated on — see the module docstring for why
    the naive (1-prob)*100 map cannot work here.
    """
    q70, q90 = np.percentile(prob_cal, [70, 90])
    return {"p_min": float(prob_cal.min()), "p_q70": float(q70),
            "p_q90": float(q90), "p_max": float(prob_cal.max())}


def prob_to_score(prob: np.ndarray, anchors: dict) -> np.ndarray:
    """Piecewise-linear, monotone decreasing in probability."""
    return np.interp(
        prob,
        [anchors["p_min"], anchors["p_q70"], anchors["p_q90"], anchors["p_max"]],
        [100.0, BAND_SOUND_MIN, BAND_NEUTRAL_MIN, 50.0],
    )


def band_of(score: np.ndarray) -> np.ndarray:
    return np.where(score >= BAND_SOUND_MIN, "sound",
                    np.where(score >= BAND_NEUTRAL_MIN, "neutral", "distress"))


def score_intervals(latent_mean: np.ndarray, latent_var: np.ndarray,
                    calibrator: LogisticRegression, anchors: dict) -> dict:
    """80% / 95% score intervals from the GP latent uncertainty.

    The latent interval is pushed through the same sigmoid + calibration +
    score map as the point estimate, so the interval stays consistent with the
    published score. Endpoints flip because score decreases in latent.
    """
    sd = np.sqrt(np.maximum(latent_var, 0.0))
    out = {}
    for tag, z in (("80", Z80), ("95", Z95)):
        hi_latent, lo_latent = latent_mean + z * sd, latent_mean - z * sd
        # high latent -> high prob -> LOW score, hence the swap
        out[f"score_lo_{tag}"] = prob_to_score(
            calibrate(calibrator, 1 / (1 + np.exp(-hi_latent))), anchors)
        out[f"score_hi_{tag}"] = prob_to_score(
            calibrate(calibrator, 1 / (1 + np.exp(-lo_latent))), anchors)
    return out
