from math import ceil, floor, trunc

import pytest

from pystatpower.core.numbers import (
    MAX_FLOAT,
    MIN_FLOAT,
    STD,
    Alpha,
    BaseNumber,
    DropOutRate,
    Interval,
    Mean,
    Percent,
    Power,
    Proportion,
    Ratio,
    Size,
)


class TestInterval:
    def test_contains(self):
        assert 0.5 in Interval(0, 1)
        assert 0 in Interval(0, 1, lower_inclusive=True)
        assert 1 in Interval(0, 1, upper_inclusive=True)
        assert 0 in Interval(0, 1, lower_inclusive=True, upper_inclusive=True)
        assert 1 in Interval(0, 1, lower_inclusive=True, upper_inclusive=True)

        with pytest.raises(TypeError):
            assert "0.5" in Interval(0, 1)

    def test_eq(self):
        assert Interval(0, 1) == Interval(0, 1)
        assert Interval(0, 1, lower_inclusive=True) == Interval(0, 1, lower_inclusive=True)
        assert Interval(0, 1, upper_inclusive=True) == Interval(0, 1, upper_inclusive=True)
        assert Interval(0, 1, lower_inclusive=True, upper_inclusive=True) == Interval(
            0, 1, lower_inclusive=True, upper_inclusive=True
        )

        # 区间范围近似相同
        assert Interval(0, 1e10) == Interval(0, 1e10 + 1)
        # 区间范围近似相同，但另一个区间包含边界
        assert Interval(0, 1e10) != Interval(0, 1e10 + 1, lower_inclusive=True)

        with pytest.raises(TypeError):
            assert Interval(0, 1) == "Interval(0, 1)"

    def test_repr(self):
        assert repr(Interval(0, 1)) == "(0, 1)"
        assert repr(Interval(0, 1, lower_inclusive=True)) == "[0, 1)"
        assert repr(Interval(0, 1, upper_inclusive=True)) == "(0, 1]"
        assert repr(Interval(0, 1, lower_inclusive=True, upper_inclusive=True)) == "[0, 1]"

    def test_pseudo_lbound(self):
        assert Interval(0, 1).pseudo_lbound() == 1e-10
        assert Interval(0, 1, lower_inclusive=True).pseudo_lbound() == 0

    def test_pseudo_ubound(self):
        assert Interval(0, 1).pseudo_ubound() == 1 - 1e-10
        assert Interval(0, 1, upper_inclusive=True).pseudo_ubound() == 1

    def test_pseudo_bound(self):
        assert Interval(0, 1).pseudo_bound() == (1e-10, 1 - 1e-10)
        assert Interval(0, 1, lower_inclusive=True).pseudo_bound() == (0, 1 - 1e-10)
        assert Interval(0, 1, upper_inclusive=True).pseudo_bound() == (1e-10, 1)
        assert Interval(0, 1, lower_inclusive=True, upper_inclusive=True).pseudo_bound() == (0, 1)


class TestNumeric:
    def test_domain(self):
        assert BaseNumber.domain == Interval(-MAX_FLOAT, MAX_FLOAT, lower_inclusive=True, upper_inclusive=True)

    def test_new(self):
        assert BaseNumber(0) == 0
        assert BaseNumber(0.5) == 0.5
        assert BaseNumber(1) == 1
        assert BaseNumber(-MAX_FLOAT) == -MAX_FLOAT
        assert BaseNumber(-MIN_FLOAT) == -MIN_FLOAT
        assert BaseNumber(MIN_FLOAT) == MIN_FLOAT
        assert BaseNumber(MAX_FLOAT) == MAX_FLOAT

        with pytest.raises(TypeError):
            BaseNumber("0.5")
        with pytest.raises(ValueError):
            BaseNumber(-MAX_FLOAT - 1)

    def test_pseudo_domain(self):
        assert BaseNumber.pseudo_bound() == (-MAX_FLOAT, MAX_FLOAT)
        assert Alpha.pseudo_bound() == (0 + MIN_FLOAT, 1 - MIN_FLOAT)
        assert Power.pseudo_bound() == (0 + MIN_FLOAT, 1 - MIN_FLOAT)
        assert Mean.pseudo_bound() == (-MAX_FLOAT + MIN_FLOAT, MAX_FLOAT - MIN_FLOAT)
        assert STD.pseudo_bound() == (0 + MIN_FLOAT, MAX_FLOAT - MIN_FLOAT)
        assert Proportion.pseudo_bound() == (0 + MIN_FLOAT, 1 - MIN_FLOAT)
        assert Percent.pseudo_bound() == (0 + MIN_FLOAT, 1 - MIN_FLOAT)
        assert Ratio.pseudo_bound() == (0 + MIN_FLOAT, MAX_FLOAT - MIN_FLOAT)
        assert Size.pseudo_bound() == (0 + MIN_FLOAT, MAX_FLOAT - MIN_FLOAT)
        assert DropOutRate.pseudo_bound() == (0, 1 - MIN_FLOAT)

    def test_operator(self):
        assert BaseNumber(1) + 1 == 2
        assert BaseNumber(1) + BaseNumber(1) == 2
        assert 1 + BaseNumber(1) == 2

        assert BaseNumber(0.2) + 0.3 == 0.5
        assert BaseNumber(0.2) + BaseNumber(0.3) == 0.5
        assert 0.2 + BaseNumber(0.3) == 0.5

        assert BaseNumber(2) - BaseNumber(1) == 1
        assert BaseNumber(2) * BaseNumber(3) == 6
        assert BaseNumber(4) / BaseNumber(2) == 2
        assert BaseNumber(5) // BaseNumber(2) == 2
        assert BaseNumber(5) % BaseNumber(2) == 1
        assert BaseNumber(2) ** BaseNumber(3) == 8

        assert abs(BaseNumber(-1)) == 1
        assert +BaseNumber(1) == 1
        assert -BaseNumber(1) == -1
        assert trunc(BaseNumber(1.5)) == 1
        assert floor(BaseNumber(1.5)) == 1
        assert ceil(BaseNumber(1.5)) == 2
        assert round(BaseNumber(3.1415)) == 3
        assert round(BaseNumber(-3.1515)) == -3
        assert round(BaseNumber(3.1415), 3) == 3.142
        assert round(BaseNumber(-3.1415), 3) == -3.142

        assert BaseNumber(1) == BaseNumber(1)
        assert BaseNumber(1) != BaseNumber(2)
        assert BaseNumber(1) < BaseNumber(2)
        assert BaseNumber(1) > BaseNumber(0)
        assert BaseNumber(1) <= BaseNumber(2)
        assert BaseNumber(1) >= BaseNumber(0)

        assert int(BaseNumber(1.2)) == 1
        assert float(BaseNumber(1.2)) == 1.2
        assert complex(BaseNumber(1.2)) == 1.2 + 0j

        assert hash(BaseNumber(3.1415)) == hash(3.1415)
        assert bool(BaseNumber(3.1415)) == bool(3.1415)

        with pytest.raises(TypeError):
            BaseNumber(1) + "1"

    def test_mix(self):
        alpha = Alpha(0.05)
        power = Power(0.8)
        mean = Mean(0)
        std = STD(10)
        proportion = Proportion(0.5)
        percent = Percent(0.5)
        ratio = Ratio(0.5)
        size = Size(7)
        dropout_rate = DropOutRate(0.5)

        assert alpha + power == 0.8 + 0.05
        assert alpha - mean == 0.05 - 0
        assert power * std == 0.8 * 10
        assert std / proportion == 10 / 0.5
        assert power // percent == 0.8 // 0.5
        assert ratio % size == 0.5 % 7
        assert std**dropout_rate == 3.1622776601683795
