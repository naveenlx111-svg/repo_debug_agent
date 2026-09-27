"""Descriptive statistics helpers."""


def mean(values):
    """Arithmetic mean; 0.0 for an empty sequence."""
    return sum(values) / len(values)


def median(values):
    if not values:
        raise ValueError("median of empty sequence")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[mid] + ordered[mid + 1]) / 2
    return ordered[mid]


def moving_average(values, window):
    """Mean of every full window of `window` consecutive values."""
    if window <= 0:
        raise ValueError("window must be positive")
    return [mean(values[i : i + window]) for i in range(len(values) - window)]


def clamp(value, low, high):
    return max(low, min(value, high))
