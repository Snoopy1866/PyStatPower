from __future__ import annotations

from dataclasses import dataclass
from math import isclose

type BaseNumberType = BaseNumber | None

#: 对于数值计算有意义的最小浮点数。
MIN_FLOAT: float = 1e-10

#: 对于数值计算有意义的最大浮点数。
MAX_FLOAT: float = 1e10


@dataclass(frozen=True)
class Interval:
    """定义一个数值区间，不支持单点区间，例如：[1, 1]。

    Parameters
    ----------
    lower : int | float
        区间的下界。
    upper : int | float
        区间的上界。
    lower_inclusive : bool
        `True` 表示包含下界，`False` 表示不包含下界。默认为 `False`。
    upper_inclusive : bool
        `True` 表示包含上界，`False` 表示不包含上界。默认为 `False`。

    Examples
    --------
    >>> itv= Interval(0, 1, lower_inclusive=True, upper_inclusive=False)
    >>> itv
    [0, 1)
    >>> 0.5 in itv
    True
    >>> 1 in itv
    False
    >>> 0 in itv
    False
    >>> itv.pseudo_bound()
    (0, 0.9999999999)
    """

    lower: int | float
    upper: int | float
    lower_inclusive: bool = False
    upper_inclusive: bool = False

    def __contains__(self, value: int | float) -> bool:
        if isinstance(value, (int, float)):
            if self.lower_inclusive:
                if self.upper_inclusive:
                    return self.lower <= value <= self.upper
                else:
                    return self.lower <= value < self.upper
            else:
                if self.upper_inclusive:
                    return self.lower < value <= self.upper
                else:
                    return self.lower < value < self.upper

        raise TypeError(f"Interval.__contains__ only supports real numbers, but you passed in a {type(value)}.")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Interval):
            return (
                isclose(self.lower, other.lower)
                and isclose(self.upper, other.upper)
                and self.lower_inclusive == other.lower_inclusive
                and self.upper_inclusive == other.upper_inclusive
            )

        raise TypeError(f"Interval.__eq__ only supports Interval, but you passed in a {type(other)}.")

    def __repr__(self) -> str:
        if self.lower_inclusive:
            if self.upper_inclusive:
                return f"[{self.lower}, {self.upper}]"
            else:
                return f"[{self.lower}, {self.upper})"
        else:
            if self.upper_inclusive:
                return f"({self.lower}, {self.upper}]"
            else:
                return f"({self.lower}, {self.upper})"

    def pseudo_lbound(self, eps: float = MIN_FLOAT) -> int | float:
        """返回区间的伪下界，用于数值计算。

        Parameters
        ----------
        eps : float
            用于计算伪下界的 ε 。默认为 :const:`MIN_FLOAT`。

        Returns
        -------
        int | float
            区间的伪下界。
        """

        if self.lower_inclusive:
            return self.lower
        else:
            return self.lower + eps

    def pseudo_ubound(self, eps: float = MIN_FLOAT) -> int | float:
        """返回区间的伪上界，用于数值计算。

        Parameters
        ----------
        eps : float
            用于计算伪上界的 ε 。默认为 :const:`MIN_FLOAT`。

        Returns
        -------
        int | float
            区间的伪上界。
        """

        if self.upper_inclusive:
            return self.upper
        else:
            return self.upper - eps

    def pseudo_bound(self, eps: float = MIN_FLOAT) -> tuple[float, float]:
        """返回区间的伪上、下界，用于数值计算。

        Parameters
        ----------
        eps : float
            用于计算伪上、下界的 ε 。默认为 :const:`MIN_FLOAT`。

        Returns
        -------
        tuple[float, float]
            区间的伪下界和伪上界。
        """

        return (self.pseudo_lbound(eps), self.pseudo_ubound(eps))


class BaseNumber(float):
    """功效分析数值类型的基类。

    - 如果传递一个 `int` 或 `float` 数值来创建一个实例，将会检查它是否在定义域 :attr:`domain` 内。
      如果不在，将会引发一个 `ValueError`，否则将返回一个新的浮点数对象，其行为与内置的浮点数对象相同。
    - 如果传递 `None` 来创建一个实例，将会返回 `None`。
    - 如果传递其他类型来创建一个实例，将会引发一个 `TypeError`。

    Examples
    --------
    >>> BaseNumber(0.5)
    0.5
    >>> BaseNumber(0.5) * BaseNumber(0.5)
    0.25
    >>> isinstance(BaseNumber(0.5), float)
    True
    """

    #: 一个 :class:`Interval` 对象，用于限制特定数值类型的取值范围。
    domain = Interval(-MAX_FLOAT, MAX_FLOAT, lower_inclusive=True, upper_inclusive=True)

    def __new__(cls, obj) -> BaseNumber | None:
        if isinstance(obj, (int, float)):
            if obj not in cls.domain:
                raise ValueError(f"{obj} is not in {cls.domain}.")
            return super().__new__(cls, obj)
        elif obj is None:
            return None
        else:
            raise TypeError(f"{obj} must be either an int, float, or None.")

    @classmethod
    def pseudo_bound(cls) -> tuple[float, float]:
        """
        返回一个元组，包含用于数值计算的伪上、下界。

        See Also
        --------
        :func:`Interval.pseudo_bound`
        """

        return cls.domain.pseudo_bound()


class Alpha(BaseNumber):
    """显著性水平"""

    #: 参见 :attr:`PowerAnalysisFloat.domain`.
    domain = Interval(0, 1)


class Power(BaseNumber):
    """检验效能"""

    #: 参见 :attr:`PowerAnalysisFloat.domain`.
    domain = Interval(0, 1)


class Mean(BaseNumber):
    """均值"""

    #: 参见 :attr:`PowerAnalysisFloat.domain`.
    domain = Interval(-MAX_FLOAT, MAX_FLOAT)


class STD(BaseNumber):
    """标准差"""

    #: 参见 :attr:`PowerAnalysisFloat.domain`.
    domain = Interval(0, MAX_FLOAT)


class Proportion(BaseNumber):
    """率"""

    #: 参见 :attr:`PowerAnalysisFloat.domain`.
    domain = Interval(0, 1)


class Percent(BaseNumber):
    """百分比"""

    #: 参见 :attr:`PowerAnalysisFloat.domain`.
    domain = Interval(0, 1)


class Ratio(BaseNumber):
    """比值"""

    #: 参见 :attr:`PowerAnalysisFloat.domain`.
    domain = Interval(0, MAX_FLOAT)


class Size(BaseNumber):
    """样本量"""

    #: 参见 :attr:`PowerAnalysisFloat.domain`.
    domain = Interval(0, MAX_FLOAT)


class DropOutRate(BaseNumber):
    """脱落率"""

    #: 参见 :attr:`PowerAnalysisFloat.domain`.
    domain = Interval(0, 1, lower_inclusive=True)


class DropOutSize(BaseNumber):
    """考虑脱落率后的样本量"""

    #: 参见 :attr:`PowerAnalysisFloat.domain`.
    domain = Interval(0, MAX_FLOAT, lower_inclusive=True)
