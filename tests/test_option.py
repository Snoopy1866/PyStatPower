from enum import Enum

import pytest

from pystatpower.core.options import Option


class TestOption:
    def test_getitem(self):
        class TestEnum(Enum, metaclass=Option):
            A = 1
            B = 2

        assert TestEnum["A"] == TestEnum.A
        assert TestEnum["a"] == TestEnum.A
        assert TestEnum["B"] == TestEnum.B
        assert TestEnum["b"] == TestEnum.B

        with pytest.raises(KeyError):
            TestEnum["C"]
        with pytest.raises(KeyError):
            TestEnum[TestEnum.A]
