import pytest

from stats import clamp, mean, median, moving_average


def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0


def test_median_odd_and_even():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5


def test_median_empty_raises():
    with pytest.raises(ValueError):
        median([])


def test_moving_average_includes_last_window():
    assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]


def test_clamp():
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0
