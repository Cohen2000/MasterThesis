"""Four-component parsing and ProfileMAE without prediction repair.

No historical failure penalties, interval schema, retry selection or repeat
aggregation are inherited. The raw response is never modified by these helpers.
"""

import json
import math

from persistence_prompt import COMPONENTS


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _values(profile):
    if not isinstance(profile, dict) or set(profile) != set(COMPONENTS):
        raise ValueError("expected exactly rho_2, rho_3, rho_4, rho_5")
    values = [profile[key] for key in COMPONENTS]
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in values):
        raise ValueError("profile components must be finite JSON numbers")
    return values


def parse_final_profile(raw_response: str) -> dict:
    """Parse the final non-empty line strictly, retaining out-of-range values."""
    lines = raw_response.strip().splitlines()
    if not lines:
        raise ValueError("empty response")
    profile = json.loads(lines[-1], object_pairs_hook=_unique_object)
    _values(profile)
    return profile


def profile_diagnostics(profile: dict) -> dict:
    values = _values(profile)
    return {
        "bounds_violation": any(value < 0 or value > 1 for value in values),
        "monotonicity_violation": any(a < b for a, b in zip(values, values[1:])),
    }


def profile_mae(prediction: dict, truth: dict) -> float:
    """Mean absolute error across the four raw finite numeric predictions."""
    predicted = _values(prediction)
    actual = _values(truth)
    if any(profile_diagnostics(truth).values()):
        raise ValueError("truth must be a bounded non-increasing profile")
    return sum(abs(a - b) for a, b in zip(predicted, actual)) / 4


def mean_occupancy(profile: dict) -> float:
    """Derived E[K/5]; no independent prediction or clipping."""
    return (1 + sum(_values(profile))) / 5


def evaluate_response(raw_response: str, truth: dict) -> dict:
    """Score one response. Caller retains every sampler seed and LLM repeat."""
    _values(truth)
    if any(profile_diagnostics(truth).values()):
        raise ValueError("truth must be a bounded non-increasing profile")
    result = {"raw_response": raw_response, "prediction": None,
              "profile_mae": None, "parse_error": None,
              "bounds_violation": None, "monotonicity_violation": None}
    try:
        prediction = parse_final_profile(raw_response)
    except ValueError as exc:
        result["parse_error"] = str(exc)
        return result
    result.update(prediction=prediction,
                  profile_mae=profile_mae(prediction, truth),
                  **profile_diagnostics(prediction))
    return result
