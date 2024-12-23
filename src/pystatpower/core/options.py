from enum import Enum, EnumMeta, unique


class Option(EnumMeta):
    """
    功效分析选项的枚举元类，可支持通过大小写不敏感的枚举值对枚举成员进行访问。

    Examples
    --------
    >>> class DayOfWeek(Enum, metaclass=Option):
    ...     SUN = 1
    ...     MON = 2
    ...     TUE = 3
    ...
    >>> DayOfWeek["sun"]
    <DayOfWeek.SUN: 1>
    >>> DayOfWeek["Mon"]
    <DayOfWeek.MON: 2>
    >>> DayOfWeek["TUE"]
    <DayOfWeek.TUE: 3>
    """

    def __getitem__(self, name):
        if isinstance(name, str):
            return super().__getitem__(name.upper())
        else:
            return super().__getitem__(name)


@unique
class Alternative(Enum, metaclass=Option):
    """备择假设类型"""

    #: 双侧检验
    TWO_SIDED = 1
    #: 单侧检验
    ONE_SIDED = 2


@unique
class SearchDirection(Enum, metaclass=Option):
    """搜索方向"""

    #: 向下搜索
    LESS = 1
    #: 向上搜索
    GREATER = 2
