"""Utilities for probability-vector majorization and T-transforms."""

from __future__ import annotations

from collections.abc import Sequence
from math import atanh, sqrt, tanh


def _descending(values: Sequence[float]) -> list[float]:
    return sorted((float(value) for value in values), reverse=True)


def is_majorized(
    x: Sequence[float],
    y: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Return whether ``x`` is majorized by ``y`` using descending order."""
    if len(x) != len(y) or not x:
        return False

    x_sorted = _descending(x)
    y_sorted = _descending(y)
    if abs(sum(x_sorted) - sum(y_sorted)) > tolerance:
        return False

    x_partial = 0.0
    y_partial = 0.0
    for x_value, y_value in zip(x_sorted[:-1], y_sorted[:-1]):
        x_partial += x_value
        y_partial += y_value
        if x_partial > y_partial + tolerance:
            return False
    return True


def t_parameters(
    x: Sequence[float],
    y: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> tuple[list[float], list[int]]:
    """Construct successive T-transform parameters for ``x`` majorized by ``y``.

    The returned indices use one-based indexing to match the research notebook.
    Inputs are copied and are never modified.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have equal lengths")
    if len(x) < 2:
        raise ValueError("x and y must contain at least two elements")
    if not is_majorized(x, y, tolerance=tolerance):
        raise ValueError("x must be majorized by y")

    x_remaining = _descending(x)
    y_remaining = _descending(y)
    parameters: list[float] = []
    indices: list[int] = []

    while len(x_remaining) > 1:
        target = x_remaining[0]
        k = next(
            (
                index
                for index in range(1, len(y_remaining))
                if y_remaining[index] - tolerance
                <= target
                <= y_remaining[index - 1] + tolerance
            ),
            None,
        )
        if k is None:
            raise ValueError("no valid T-transform index was found")

        denominator = y_remaining[0] - y_remaining[k]
        if abs(denominator) <= tolerance:
            parameter = 1.0
        else:
            parameter = (target - y_remaining[k]) / denominator
        if not -tolerance <= parameter <= 1.0 + tolerance:
            raise ValueError("computed T-transform parameter lies outside [0, 1]")
        parameter = min(1.0, max(0.0, parameter))

        first = y_remaining[0]
        kth = y_remaining[k]
        y_remaining[k] = (1.0 - parameter) * first + parameter * kth
        parameters.append(parameter)
        indices.append(k + 1)

        x_remaining = x_remaining[1:]
        y_remaining = y_remaining[1:]

    return parameters, indices


def truncated_tmss_probabilities(
    squeezing: float,
    dimension: int,
    *,
    renormalize: bool = False,
) -> list[float]:
    """Return a truncated two-mode-squeezed-state probability vector."""
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    minimum_squeezing = atanh(1.0 / sqrt(2.0))
    if squeezing < minimum_squeezing:
        raise ValueError("squeezing is below the Nielsen-criterion threshold")

    coefficient = tanh(squeezing)
    values = [
        (1.0 - coefficient**2) * coefficient ** (2 * index)
        for index in range(dimension)
    ]
    if renormalize:
        total = sum(values)
        values = [value / total for value in values]
    return values
