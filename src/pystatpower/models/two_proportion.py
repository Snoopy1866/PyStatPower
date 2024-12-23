"""两独立样本差异性检验"""

from enum import Enum, Flag, auto, unique
from math import ceil, sqrt

from scipy.optimize import brentq
from scipy.stats import norm

from pystatpower.core.numbers import (
    Alpha,
    DropOutRate,
    DropOutSize,
    Interval,
    Percent,
    Power,
    Proportion,
    Ratio,
    Size,
)
from pystatpower.core.options import Alternative, Option, SearchDirection

__all__ = [
    "GroupAllocation",
    "solve_for_sample_size",
    "solve_for_alpha",
    "solve_for_power",
    "solve_for_treatment_proportion",
    "solve_for_reference_proportion",
]


@unique
class Target(Enum, metaclass=Option):
    """求解目标

    Attributes
    __________
        SIZE : (int)
            样本量
        ALPHA : (int)
            显著性水平
        POWER : (int)
            检验效能
        TREATMENT_PROPORTION : (int)
            试验组的率
        REFERENCE_PROPORTION : (int)
            对照组的率
    """

    SIZE = 1
    ALPHA = 2
    POWER = 3
    TREATMENT_PROPORTION = 4
    REFERENCE_PROPORTION = 5


@unique
class TestType(Enum, metaclass=Option):
    """检验类型

    Attributes
    ----------
        Z_TEST_POOLED : (int)
            Z 检验（合并方差）
        Z_TEST_UNPOOLED : (int)
            Z 检验（独立方差）
        Z_TEST_CC_POOLED : (int)
            Z 检验（连续性校正，合并方差）
        Z_TEST_CC_UNPOOLED : (int)
            Z 检验（连续性校正，独立方差）
    """

    Z_TEST_POOLED = 1
    Z_TEST_UNPOOLED = 2
    Z_TEST_CC_POOLED = 3
    Z_TEST_CC_UNPOOLED = 4


class GroupAllocationOption(Flag, metaclass=Option):
    """样本量分配类型（求解目标：样本量）

    Attributes
    ----------
        EQUAL
            等量分配
        SIZE_OF_TOTAL
            总样本量
        SIZE_OF_EACH
            单组样本量
        SIZE_OF_TREATMENT
            试验组样本量
        SIZE_OF_REFERENCE
            对照组样本量
        RATIO_OF_TREATMENT_TO_REFERENCE
            试验组与对照组样本量比例
        RATIO_OF_REFERENCE_TO_TREATMENT
            对照组与试验组样本量比例
        PERCENT_OF_TREATMENT
            试验组样本百分比
        PERCENT_OF_REFERENCE
            对照组样本百分比
    """

    EQUAL = auto()
    SIZE_OF_TOTAL = auto()
    SIZE_OF_EACH = auto()
    SIZE_OF_TREATMENT = auto()
    SIZE_OF_REFERENCE = auto()
    RATIO_OF_TREATMENT_TO_REFERENCE = auto()
    RATIO_OF_REFERENCE_TO_TREATMENT = auto()
    PERCENT_OF_TREATMENT = auto()
    PERCENT_OF_REFERENCE = auto()


class GroupAllocation:
    """用于定义样本量分配方式的类"""

    def __init__(
        self,
        size_of_total: float = None,
        size_of_each: float = None,
        size_of_treatment: float = None,
        size_of_reference: float = None,
        ratio_of_treatment_to_reference: float = None,
        ratio_of_reference_to_treatment: float = None,
        percent_of_treatment: float = None,
        percent_of_reference: float = None,
    ):
        """初始化 GroupAllocation 类

        Args:
            size_of_total (float, optional): 总样本量
            size_of_each (float, optional): 单组样本量
            size_of_treatment (float, optional): 试验组样本量
            size_of_reference (float, optional): 对照组样本量
            ratio_of_treatment_to_reference (float, optional): 试验组样本量与对照组样本量的比值
            ratio_of_reference_to_treatment (float, optional): 对照组样本量与试验组样本量的比值
            percent_of_treatment (float, optional): 试验组样本量占总样本量的百分比
            percent_of_reference (float, optional): 对照组样本量占总样本量的百分比
        """

        self.group_allocation_params_dict_for_size = dict(
            size_of_treatment=Size(size_of_treatment),
            size_of_reference=Size(size_of_reference),
            ratio_of_treatment_to_reference=Ratio(ratio_of_treatment_to_reference),
            ratio_of_reference_to_treatment=Ratio(ratio_of_reference_to_treatment),
            percent_of_treatment=Percent(percent_of_treatment),
            percent_of_reference=Percent(percent_of_reference),
        )

        self.group_allocation_params_dict_for_not_size = self.group_allocation_params_dict_for_size | dict(
            size_of_total=Size(size_of_total),
            size_of_each=Size(size_of_each),
        )

    def get_group_allocation_for_target(self, target: Target):
        match target:
            case Target.SIZE:
                return GroupAllocationForSize(**self.group_allocation_params_dict_for_size)
            case Target.ALPHA:
                return GroupAllocationForAlpha(**self.group_allocation_params_dict_for_not_size)
            case Target.POWER:
                return GroupAllocationForPower(**self.group_allocation_params_dict_for_not_size)
            case Target.TREATMENT_PROPORTION:
                return GroupAllocationForTreatmentProportion(**self.group_allocation_params_dict_for_not_size)
            case Target.REFERENCE_PROPORTION:
                return GroupAllocationForReferenceProportion(**self.group_allocation_params_dict_for_not_size)
            case _:  # pragma: no cover
                assert False, "不支持的求解目标"


class GroupAllocationForSize:
    """用于定义样本量分配方式的类（求解目标: 样本量）"""

    def __init__(
        self,
        size_of_treatment: float = None,
        size_of_reference: float = None,
        ratio_of_treatment_to_reference: float = None,
        ratio_of_reference_to_treatment: float = None,
        percent_of_treatment: float = None,
        percent_of_reference: float = None,
    ):
        """初始化 GroupAllocation.ForSize 类

        Args:
            size_of_treatment (float, optional): 试验组样本量
            size_of_reference (float, optional): 对照组样本量
            ratio_of_treatment_to_reference (float, optional): 试验组样本量与对照组样本量的比值
            ratio_of_reference_to_treatment (float, optional): 对照组样本量与试验组样本量的比值
            percent_of_treatment (float, optional): 试验组样本量占总样本量的百分比
            percent_of_reference (float, optional): 对照组样本量占总样本量的百分比

        Raises:
            ValueError: 如果指定了不支持的分配方式
        """

        self.size_of_treatment = Size(size_of_treatment)
        self.size_of_reference = Size(size_of_reference)
        self.ratio_of_treatment_to_reference = Ratio(ratio_of_treatment_to_reference)
        self.ratio_of_reference_to_treatment = Ratio(ratio_of_reference_to_treatment)
        self.percent_of_treatment = Percent(percent_of_treatment)
        self.percent_of_reference = Percent(percent_of_reference)

        kwargs = {
            "size_of_treatment": self.size_of_treatment,
            "size_of_reference": self.size_of_reference,
            "ratio_of_treatment_to_reference": self.ratio_of_treatment_to_reference,
            "ratio_of_reference_to_treatment": self.ratio_of_reference_to_treatment,
            "percent_of_treatment": self.percent_of_treatment,
            "percent_of_reference": self.percent_of_reference,
        }

        kwargs_spec = {k: v for k, v in kwargs.items() if v is not None}

        if len(kwargs_spec) == 0:
            self.group_allocation_option = GroupAllocationOption.EQUAL
        elif len(kwargs_spec) == 1:
            arg = list(kwargs_spec.keys())[0]
            self.group_allocation_option = GroupAllocationOption[arg]
        else:
            raise ValueError("不支持的参数")

        match self.group_allocation_option:
            case GroupAllocationOption.EQUAL:
                self.treatment_size_formula = lambda n: n
                self.reference_size_formula = lambda n: n
            case GroupAllocationOption.SIZE_OF_TREATMENT:
                self.treatment_size_formula = lambda n: size_of_treatment
                self.reference_size_formula = lambda n: n
            case GroupAllocationOption.SIZE_OF_REFERENCE:
                self.treatment_size_formula = lambda n: n
                self.reference_size_formula = lambda n: size_of_reference
            case GroupAllocationOption.RATIO_OF_TREATMENT_TO_REFERENCE:
                self.treatment_size_formula = lambda n: ratio_of_treatment_to_reference * n
                self.reference_size_formula = lambda n: n
            case GroupAllocationOption.RATIO_OF_REFERENCE_TO_TREATMENT:
                self.treatment_size_formula = lambda n: n
                self.reference_size_formula = lambda n: ratio_of_reference_to_treatment * n
            case GroupAllocationOption.PERCENT_OF_TREATMENT:
                self.treatment_size_formula = lambda n: n
                self.reference_size_formula = lambda n: (1 - percent_of_treatment) / percent_of_treatment * n
            case GroupAllocationOption.PERCENT_OF_REFERENCE:
                self.treatment_size_formula = lambda n: (1 - percent_of_reference) / percent_of_reference * n
                self.reference_size_formula = lambda n: n
            case _:  # pragma: no cover
                assert False, "不支持的参数"


class GroupAllocationForNotSize:
    def __init__(
        self,
        size_of_total: float = None,
        size_of_each: float = None,
        size_of_treatment: float = None,
        size_of_reference: float = None,
        ratio_of_treatment_to_reference: float = None,
        ratio_of_reference_to_treatment: float = None,
        percent_of_treatment: float = None,
        percent_of_reference: float = None,
    ):
        """初始化 GroupAllocation.ForNotSize 类

        Args:
            size_of_total (float, optional): 总样本量
            size_of_each (float, optional): 单组样本量
            size_of_treatment (float, optional): 试验组样本量
            size_of_reference (float, optional): 对照组样本量
            ratio_of_treatment_to_reference (float, optional): 试验组样本量与对照组样本量的比值
            ratio_of_reference_to_treatment (float, optional): 对照组样本量与试验组样本量的比值
            percent_of_treatment (float, optional): 试验组样本量占总样本量的百分比
            percent_of_reference (float, optional): 对照组样本量占总样本量的百分比

        Raises:
            ValueError: 如果指定了不支持的分配方式
        """

        self.size_of_total = Size(size_of_total)
        self.size_of_each = Size(size_of_each)
        self.size_of_treatment = Size(size_of_treatment)
        self.size_of_reference = Size(size_of_reference)
        self.ratio_of_treatment_to_reference = Ratio(ratio_of_treatment_to_reference)
        self.ratio_of_reference_to_treatment = Ratio(ratio_of_reference_to_treatment)
        self.percent_of_treatment = Percent(percent_of_treatment)
        self.percent_of_reference = Percent(percent_of_reference)

        kwargs = {
            "size_of_total": self.size_of_total,
            "size_of_each": self.size_of_each,
            "size_of_treatment": self.size_of_treatment,
            "size_of_reference": self.size_of_reference,
            "ratio_of_treatment_to_reference": self.ratio_of_treatment_to_reference,
            "ratio_of_reference_to_treatment": self.ratio_of_reference_to_treatment,
            "percent_of_treatment": self.percent_of_treatment,
            "percent_of_reference": self.percent_of_reference,
        }

        kwargs_spec = {k: v for k, v in kwargs.items() if v is not None}
        if len(kwargs_spec) == 1:
            arg = list(kwargs_spec.keys())[0]
            if arg in ["size_of_total", "size_of_each", "size_of_treatment", "size_of_reference"]:
                self.group_allocation_option = GroupAllocationOption.EQUAL | GroupAllocationOption[arg]
            else:
                raise ValueError(
                    "参数不足，若采用等量分配方式，则指定的参数必须是 'size_of_total', 'size_of_each', 'size_of_treatment' 或 'size_of_reference' 其中之一"
                )
        elif len(kwargs_spec) == 2:
            args = list(kwargs_spec.keys())
            self.group_allocation_option = GroupAllocationOption[args[0]] | GroupAllocationOption[args[1]]
        else:
            raise ValueError("不支持的参数组合")

        match self.group_allocation_option:
            case option if option == GroupAllocationOption.EQUAL | GroupAllocationOption.SIZE_OF_TOTAL:
                self.treatment_size_formula = lambda: size_of_total / 2
                self.reference_size_formula = lambda: size_of_total / 2
            case option if option == GroupAllocationOption.EQUAL | GroupAllocationOption.SIZE_OF_EACH:
                self.treatment_size_formula = lambda: size_of_each
                self.reference_size_formula = lambda: size_of_each
            case option if option == GroupAllocationOption.EQUAL | GroupAllocationOption.SIZE_OF_TREATMENT:
                self.treatment_size_formula = lambda: size_of_treatment
                self.reference_size_formula = lambda: size_of_treatment
            case option if option == GroupAllocationOption.EQUAL | GroupAllocationOption.SIZE_OF_REFERENCE:
                self.treatment_size_formula = lambda: size_of_reference
                self.reference_size_formula = lambda: size_of_reference
            case option if option == GroupAllocationOption.SIZE_OF_TOTAL | GroupAllocationOption.SIZE_OF_TREATMENT:
                self.treatment_size_formula = lambda: size_of_treatment
                self.reference_size_formula = lambda: size_of_total - size_of_treatment
            case option if option == GroupAllocationOption.SIZE_OF_TOTAL | GroupAllocationOption.SIZE_OF_REFERENCE:
                self.treatment_size_formula = lambda: size_of_total - size_of_reference
                self.reference_size_formula = lambda: size_of_reference
            case (
                option
            ) if option == GroupAllocationOption.SIZE_OF_TOTAL | GroupAllocationOption.RATIO_OF_TREATMENT_TO_REFERENCE:
                self.treatment_size_formula = (
                    lambda: size_of_total * ratio_of_treatment_to_reference / (1 + ratio_of_treatment_to_reference)
                )
                self.reference_size_formula = lambda: size_of_total / (1 + ratio_of_treatment_to_reference)
            case (
                option
            ) if option == GroupAllocationOption.SIZE_OF_TOTAL | GroupAllocationOption.RATIO_OF_REFERENCE_TO_TREATMENT:
                self.treatment_size_formula = lambda: size_of_total / (1 + ratio_of_reference_to_treatment)
                self.reference_size_formula = lambda: (
                    size_of_total * ratio_of_reference_to_treatment / (1 + ratio_of_reference_to_treatment)
                )
            case option if option == GroupAllocationOption.SIZE_OF_TOTAL | GroupAllocationOption.PERCENT_OF_TREATMENT:
                self.treatment_size_formula = lambda: size_of_total * percent_of_treatment
                self.reference_size_formula = lambda: size_of_total * (1 - percent_of_treatment)
            case option if option == GroupAllocationOption.SIZE_OF_TOTAL | GroupAllocationOption.PERCENT_OF_REFERENCE:
                self.treatment_size_formula = lambda: size_of_total * (1 - percent_of_reference)
                self.reference_size_formula = lambda: size_of_total * percent_of_reference
            case option if option == GroupAllocationOption.SIZE_OF_TREATMENT | GroupAllocationOption.SIZE_OF_REFERENCE:
                self.treatment_size_formula = lambda: size_of_treatment
                self.reference_size_formula = lambda: size_of_reference
            case (
                option
            ) if option == GroupAllocationOption.SIZE_OF_TREATMENT | GroupAllocationOption.RATIO_OF_TREATMENT_TO_REFERENCE:
                self.treatment_size_formula = lambda: size_of_treatment
                self.reference_size_formula = lambda: size_of_treatment / ratio_of_treatment_to_reference
            case (
                option
            ) if option == GroupAllocationOption.SIZE_OF_TREATMENT | GroupAllocationOption.RATIO_OF_REFERENCE_TO_TREATMENT:
                self.treatment_size_formula = lambda: size_of_treatment
                self.reference_size_formula = lambda: size_of_treatment * ratio_of_reference_to_treatment
            case (
                option
            ) if option == GroupAllocationOption.SIZE_OF_TREATMENT | GroupAllocationOption.PERCENT_OF_TREATMENT:
                self.treatment_size_formula = lambda: size_of_treatment
                self.reference_size_formula = (
                    lambda: size_of_treatment * (1 - percent_of_treatment) / percent_of_treatment
                )
            case (
                option
            ) if option == GroupAllocationOption.SIZE_OF_TREATMENT | GroupAllocationOption.PERCENT_OF_REFERENCE:
                self.treatment_size_formula = lambda: size_of_treatment
                self.reference_size_formula = (
                    lambda: size_of_treatment * percent_of_reference / (1 - percent_of_reference)
                )
            case (
                option
            ) if option == GroupAllocationOption.SIZE_OF_REFERENCE | GroupAllocationOption.RATIO_OF_TREATMENT_TO_REFERENCE:
                self.treatment_size_formula = lambda: size_of_reference * ratio_of_treatment_to_reference
                self.reference_size_formula = lambda: size_of_reference
            case (
                option
            ) if option == GroupAllocationOption.SIZE_OF_REFERENCE | GroupAllocationOption.RATIO_OF_REFERENCE_TO_TREATMENT:
                self.treatment_size_formula = lambda: size_of_reference / ratio_of_reference_to_treatment
                self.reference_size_formula = lambda: size_of_reference
            case (
                option
            ) if option == GroupAllocationOption.SIZE_OF_REFERENCE | GroupAllocationOption.PERCENT_OF_TREATMENT:
                self.treatment_size_formula = (
                    lambda: size_of_reference * percent_of_treatment / (1 - percent_of_treatment)
                )
                self.reference_size_formula = lambda: size_of_reference
            case (
                option
            ) if option == GroupAllocationOption.SIZE_OF_REFERENCE | GroupAllocationOption.PERCENT_OF_REFERENCE:
                self.treatment_size_formula = (
                    lambda: size_of_reference * (1 - percent_of_reference) / percent_of_reference
                )
                self.reference_size_formula = lambda: size_of_reference
            case _:
                raise ValueError("不支持的参数组合")


class GroupAllocationForAlpha(GroupAllocationForNotSize):
    """用于定义样本量分配方式的类（求解目标: 显著性水平）"""

    pass


class GroupAllocationForPower(GroupAllocationForNotSize):
    """用于定义样本量分配方式的类（求解目标: 检验效能）"""

    pass


class GroupAllocationForTreatmentProportion(GroupAllocationForNotSize):
    """用于定义样本量分配方式的类（求解目标: 试验组的率）"""

    pass


class GroupAllocationForReferenceProportion(GroupAllocationForNotSize):
    """用于定义样本量分配方式的类（求解目标: 对照组的率）"""

    pass


def fun_power(
    alpha: float,
    treatment_n: float,
    reference_n: float,
    treatment_proportion: float,
    reference_proportion: float,
    alternative: Alternative,
    test_type: TestType,
):
    """_summary_

    Args:
        alpha (float): 显著性水平
        treatment_n (float): 试验组样本量
        reference_n (float): 对照组样本量
        treatment_proportion (float): 试验组的率
        reference_proportion (float): 对照组的率
        alternative (Alternative): 备择假设类型
        test_type (TestType): 检验类型

    Returns:
        power (float): 检验效能
    """
    n1 = treatment_n
    n2 = reference_n
    p1 = treatment_proportion
    p2 = reference_proportion
    q1 = 1 - p1
    q2 = 1 - p2

    p_bar = (n1 * p1 + n2 * p2) / (n1 + n2)
    q_bar = 1 - p_bar

    se_p = sqrt(p_bar * q_bar * (1 / n1 + 1 / n2))
    se_u = sqrt(p1 * q1 / n1 + p2 * q2 / n2)

    # 计算标准误
    match test_type:
        case TestType.Z_TEST_POOLED | TestType.Z_TEST_CC_POOLED:
            se = se_p
        case TestType.Z_TEST_UNPOOLED | TestType.Z_TEST_CC_UNPOOLED:
            se = se_u
        case _:  # pragma: no cover
            assert False, "未知的检验类型"

    # 连续性校正
    c = 0
    if test_type in [TestType.Z_TEST_CC_POOLED, TestType.Z_TEST_CC_UNPOOLED]:
        c = (1 / 2) * (1 / n1 + 1 / n2)

    # 计算检验效能
    match alternative:
        case Alternative.TWO_SIDED:
            z_alpha = norm.ppf(1 - alpha / 2)
            power = (
                norm.cdf((-z_alpha * se - c - (p1 - p2)) / se_u) + 1 - norm.cdf((z_alpha * se + c - (p1 - p2)) / se_u)
            )
        case Alternative.ONE_SIDED:
            z_alpha = norm.ppf(1 - alpha)
            if p1 > p2:
                power = 1 - norm.cdf((z_alpha * se + c - (p1 - p2)) / se_u)
            elif p1 <= p2:
                power = norm.cdf((-z_alpha * se - c - (p1 - p2)) / se_u)
        case _:  # pragma: no cover
            assert False, "未知的备择假设类型"

    return power


class TwoProportion:
    """两独立样本差异性功效分析模型"""

    class ForSize:
        """两独立样本差异性功效分析模型（求解目标: 样本量）"""

        def __init__(
            self,
            alpha: Alpha,
            power: Power,
            treatment_proportion: Proportion,
            reference_proportion: Proportion,
            alternative: Alternative,
            test_type: TestType,
            group_allocation: GroupAllocationForSize,
            dropout_rate: DropOutRate,
        ):
            self.alpha = alpha
            self.power = power
            self.treatment_proportion = treatment_proportion
            self.reference_proportion = reference_proportion
            self.alternative = alternative
            self.test_type = test_type
            self.group_allocation = group_allocation
            self.dropout_rate = dropout_rate

        def solve(self):
            self._eval = (
                lambda n: fun_power(
                    self.alpha,
                    self.group_allocation.treatment_size_formula(n),
                    self.group_allocation.reference_size_formula(n),
                    self.treatment_proportion,
                    self.reference_proportion,
                    self.alternative,
                    self.test_type,
                )
                - self.power
            )

            lbound, ubound = Size.pseudo_bound()
            try:
                n = brentq(self._eval, lbound, ubound)
            except ValueError as e:
                raise ValueError("无解") from e

            self.treatment_size = Size(ceil(self.group_allocation.treatment_size_formula(n)))
            self.reference_size = Size(ceil(self.group_allocation.reference_size_formula(n)))

            self.treatment_size_include_dropouts = Size(ceil(self.treatment_size / (1 - self.dropout_rate)))
            self.reference_size_include_dropouts = Size(ceil(self.reference_size / (1 - self.dropout_rate)))
            self.treatment_dropouts = DropOutSize(self.treatment_size_include_dropouts - self.treatment_size)
            self.reference_dropouts = DropOutSize(self.reference_size_include_dropouts - self.reference_size)

    class ForAlpha:
        """两独立样本差异性功效分析模型（求解目标: 显著性水平）"""

        def __init__(
            self,
            power: Power,
            treatment_proportion: Proportion,
            reference_proportion: Proportion,
            alternative: Alternative,
            test_type: TestType,
            group_allocation: GroupAllocationForAlpha,
            dropout_rate: DropOutRate,
        ):
            self.power = power
            self.treatment_proportion = treatment_proportion
            self.reference_proportion = reference_proportion
            self.alternative = alternative
            self.test_type = test_type
            self.group_allocation = group_allocation
            self.dropout_rate = dropout_rate

        def solve(self):
            self._eval = (
                lambda alpha: fun_power(
                    alpha,
                    self.group_allocation.treatment_size_formula(),
                    self.group_allocation.reference_size_formula(),
                    self.treatment_proportion,
                    self.reference_proportion,
                    self.alternative,
                    self.test_type,
                )
                - self.power
            )

            lbound, ubound = Alpha.pseudo_bound()
            try:
                alpha = brentq(self._eval, lbound, ubound)
            except ValueError as e:  # pragma: no cover
                raise ValueError("无解") from e

            self.alpha = Alpha(alpha)

            self.treatment_size = Size(ceil(self.group_allocation.treatment_size_formula()))
            self.reference_size = Size(ceil(self.group_allocation.reference_size_formula()))

            self.treatment_size_include_dropouts = Size(ceil(self.treatment_size / (1 - self.dropout_rate)))
            self.reference_size_include_dropouts = Size(ceil(self.reference_size / (1 - self.dropout_rate)))
            self.treatment_dropouts = DropOutSize(self.treatment_size_include_dropouts - self.treatment_size)
            self.reference_dropouts = DropOutSize(self.reference_size_include_dropouts - self.reference_size)

    class ForPower:
        """两独立样本差异性功效分析模型（求解目标: 检验效能）"""

        def __init__(
            self,
            alpha: Alpha,
            treatment_proportion: Proportion,
            reference_proportion: Proportion,
            alternative: Alternative,
            test_type: TestType,
            group_allocation: GroupAllocationForPower,
            dropout_rate: DropOutRate,
        ):
            self.alpha = alpha
            self.treatment_proportion = treatment_proportion
            self.reference_proportion = reference_proportion
            self.alternative = alternative
            self.test_type = test_type
            self.group_allocation = group_allocation
            self.dropout_rate = dropout_rate

        def solve(self):
            power = fun_power(
                self.alpha,
                self.group_allocation.treatment_size_formula(),
                self.group_allocation.reference_size_formula(),
                self.treatment_proportion,
                self.reference_proportion,
                self.alternative,
                self.test_type,
            )

            self.power = Power(power)

            self.treatment_size = Size(ceil(self.group_allocation.treatment_size_formula()))
            self.reference_size = Size(ceil(self.group_allocation.reference_size_formula()))

            self.treatment_size_include_dropouts = Size(ceil(self.treatment_size / (1 - self.dropout_rate)))
            self.reference_size_include_dropouts = Size(ceil(self.reference_size / (1 - self.dropout_rate)))
            self.treatment_dropouts = DropOutSize(self.treatment_size_include_dropouts - self.treatment_size)
            self.reference_dropouts = DropOutSize(self.reference_size_include_dropouts - self.reference_size)

    class ForTreatmentProportion:
        """两独立样本差异性功效分析模型（求解目标: 试验组的率）"""

        def __init__(
            self,
            alpha: Alpha,
            power: Power,
            reference_proportion: Proportion,
            alternative: Alternative,
            test_type: TestType,
            group_allocation: GroupAllocationForTreatmentProportion,
            search_direction: SearchDirection,
            dropout_rate: DropOutRate,
        ):
            self.alpha = alpha
            self.power = power
            self.reference_proportion = reference_proportion
            self.alternative = alternative
            self.test_type = test_type
            self.group_allocation = group_allocation
            self.search_direction = search_direction
            self.dropout_rate = dropout_rate

        def solve(self):
            self._eval = (
                lambda treatment_proportion: fun_power(
                    self.alpha,
                    self.group_allocation.treatment_size_formula(),
                    self.group_allocation.reference_size_formula(),
                    treatment_proportion,
                    self.reference_proportion,
                    self.alternative,
                    self.test_type,
                )
                - self.power
            )

            match self.search_direction:
                case SearchDirection.LESS:
                    lbound, ubound = Interval(0, self.reference_proportion).pseudo_bound()
                    try:
                        treatment_proportion = brentq(self._eval, lbound, ubound)
                    except ValueError as e:  # pragma: no cover
                        raise ValueError("无解") from e
                case SearchDirection.GREATER:
                    lbound, ubound = Interval(self.reference_proportion, 1).pseudo_bound()
                    try:
                        treatment_proportion = brentq(self._eval, lbound, ubound)
                    except ValueError as e:
                        raise ValueError("无解") from e
                case _:  # pragma: no cover
                    assert False, "未知的搜索方向"

            self.treatment_proportion = Proportion(treatment_proportion)

            self.treatment_size = Size(ceil(self.group_allocation.treatment_size_formula()))
            self.reference_size = Size(ceil(self.group_allocation.reference_size_formula()))

            self.treatment_size_include_dropouts = Size(ceil(self.treatment_size / (1 - self.dropout_rate)))
            self.reference_size_include_dropouts = Size(ceil(self.reference_size / (1 - self.dropout_rate)))
            self.treatment_dropouts = DropOutSize(self.treatment_size_include_dropouts - self.treatment_size)
            self.reference_dropouts = DropOutSize(self.reference_size_include_dropouts - self.reference_size)

    class ForReferenceProportion:
        """两独立样本差异性功效分析模型（求解目标: 对照组的率）"""

        def __init__(
            self,
            alpha: Alpha,
            power: Power,
            treatment_proportion: Proportion,
            alternative: Alternative,
            test_type: TestType,
            group_allocation: GroupAllocationForReferenceProportion,
            search_direction: SearchDirection,
            dropout_rate: DropOutRate,
        ):
            self.alpha = alpha
            self.power = power
            self.treatment_proportion = treatment_proportion
            self.alternative = alternative
            self.test_type = test_type
            self.group_allocation = group_allocation
            self.search_direction = search_direction
            self.dropout_rate = dropout_rate

        def solve(self):
            self._eval = (
                lambda reference_proportion: fun_power(
                    self.alpha,
                    self.group_allocation.treatment_size_formula(),
                    self.group_allocation.reference_size_formula(),
                    self.treatment_proportion,
                    reference_proportion,
                    self.alternative,
                    self.test_type,
                )
                - self.power
            )
            match self.search_direction:
                case SearchDirection.LESS:
                    lbound, ubound = Interval(0, self.treatment_proportion).pseudo_bound()
                    try:
                        reference_proportion = brentq(self._eval, lbound, ubound)
                    except ValueError as e:  # pragma: no cover
                        raise ValueError("无解") from e
                case SearchDirection.GREATER:
                    lbound, ubound = Interval(self.treatment_proportion, 1).pseudo_bound()
                    try:
                        reference_proportion = brentq(self._eval, lbound, ubound)
                    except ValueError as e:
                        raise ValueError("无解") from e
                case _:  # pragma: no cover
                    assert False, "未知的搜索方向"

            self.reference_proportion = Proportion(reference_proportion)

            self.treatment_size = Size(ceil(self.group_allocation.treatment_size_formula()))
            self.reference_size = Size(ceil(self.group_allocation.reference_size_formula()))

            self.treatment_size_include_dropouts = Size(ceil(self.treatment_size / (1 - self.dropout_rate)))
            self.reference_size_include_dropouts = Size(ceil(self.reference_size / (1 - self.dropout_rate)))
            self.treatment_dropouts = DropOutSize(self.treatment_size_include_dropouts - self.treatment_size)
            self.reference_dropouts = DropOutSize(self.reference_size_include_dropouts - self.reference_size)


def solve_for_sample_size(
    alpha: float,
    power: float,
    treatment_proportion: float,
    reference_proportion: float,
    alternative: str,
    test_type: str,
    group_allocation: GroupAllocation = GroupAllocation(),
    dropout_rate: float = 0,
    full_output: bool = False,
):
    """求解样本量

    Args:
        alpha (float): 显著性水平
        power (float): 检验效能
        treatment_proportion (float): 试验组的率
        reference_proportion (float): 对照组的率
        alternative (str): 备择假设类型
        test_type (str): 检验类型
        group_allocation (GroupAllocation.ForSize, optional): 样本量分配模式。默认值: GroupAllocation.ForSize(GroupAllocationOption.EQUAL)
        dropout_rate (float, optional): 退出率。默认值: 0
        full_output (bool, optional): 是否输出完整结果。默认值: False
    """

    model = TwoProportion.ForSize(
        alpha=Alpha(alpha),
        power=Power(power),
        alternative=Alternative[alternative],
        test_type=TestType[test_type],
        treatment_proportion=Proportion(treatment_proportion),
        reference_proportion=Proportion(reference_proportion),
        group_allocation=group_allocation.get_group_allocation_for_target(Target.SIZE),
        dropout_rate=DropOutRate(dropout_rate),
    )
    model.solve()

    if full_output:
        return model
    return model.treatment_size, model.reference_size


def solve_for_alpha(
    power: float,
    treatment_proportion: float,
    reference_proportion: float,
    alternative: str,
    test_type: str,
    group_allocation: GroupAllocation,
    dropout_rate: float = 0,
    full_output: bool = False,
):
    """求解显著性水平

    Args:
        power (float): 检验效能
        treatment_proportion (float): 试验组的率
        reference_proportion (float): 对照组的率
        alternative (str): 备择假设类型，可选值: "TWO_SIDED", "ONE_SIDED"
        test_type (str): 检验类型，可选值: "Z_TEST_POOLED", "Z_TEST_UNPOOLED", "Z_TEST_CC_POOLED", "Z_TEST_CC_UNPOOLED"
        group_allocation (GroupAllocation.ForSize, optional): 样本量分配模式
        dropout_rate (float, optional): 退出率。默认值: 0
        full_output (bool, optional): 是否输出完整结果。默认值: False
    """

    model = TwoProportion.ForAlpha(
        power=Power(power),
        alternative=Alternative[alternative],
        test_type=TestType[test_type],
        treatment_proportion=Proportion(treatment_proportion),
        reference_proportion=Proportion(reference_proportion),
        group_allocation=group_allocation.get_group_allocation_for_target(Target.ALPHA),
        dropout_rate=DropOutRate(dropout_rate),
    )
    model.solve()

    if full_output:
        return model
    return model.alpha


def solve_for_power(
    alpha: float,
    treatment_proportion: float,
    reference_proportion: float,
    alternative: str,
    test_type: str,
    group_allocation: GroupAllocation,
    dropout_rate: float = 0,
    full_output: bool = False,
):
    """求解检验效能

    Args:
        alpha (float): 显著性水平
        treatment_proportion (float): 试验组的率
        reference_proportion (float): 对照组的率
        alternative (str): 备择假设类型，可选值: "TWO_SIDED", "ONE_SIDED"
        test_type (str): 检验类型，可选值: "Z_TEST_POOLED", "Z_TEST_UNPOOLED", "Z_TEST_CC_POOLED", "Z_TEST_CC_UNPOOLED"
        group_allocation (GroupAllocation.ForSize, optional): 样本量分配模式
        dropout_rate (float, optional): 退出率。默认值: 0
        full_output (bool, optional): 是否输出完整结果。默认值: False
    """

    model = TwoProportion.ForPower(
        alpha=Alpha(alpha),
        alternative=Alternative[alternative],
        test_type=TestType[test_type],
        treatment_proportion=Proportion(treatment_proportion),
        reference_proportion=Proportion(reference_proportion),
        group_allocation=group_allocation.get_group_allocation_for_target(Target.POWER),
        dropout_rate=DropOutRate(dropout_rate),
    )
    model.solve()

    if full_output:
        return model
    return model.power


def solve_for_treatment_proportion(
    alpha: float,
    power: float,
    reference_proportion: float,
    alternative: str,
    test_type: str,
    group_allocation: GroupAllocation,
    search_direction: SearchDirection,
    dropout_rate: float = 0,
    full_output: bool = False,
):
    """求解试验组的率

    Args:
        alpha (float): 显著性水平
        power (float): 检验效能
        reference_proportion (float): 对照组的率
        alternative (str): 备择假设类型，可选值: "TWO_SIDED", "ONE_SIDED"
        test_type (str): 检验类型，可选值: "Z_TEST_POOLED", "Z_TEST_UNPOOLED", "Z_TEST_CC_POOLED", "Z_TEST_CC_UNPOOLED"
        group_allocation (GroupAllocation.ForSize, optional): 样本量分配模式
        search_direction (str): 搜索方向，可选值: "LESS", "GREATER"
        dropout_rate (float, optional): 退出率。默认值: 0
        full_output (bool, optional): 是否输出完整结果。默认值: False
    """

    model = TwoProportion.ForTreatmentProportion(
        alpha=Alpha(alpha),
        power=Power(power),
        alternative=Alternative[alternative],
        test_type=TestType[test_type],
        reference_proportion=Proportion(reference_proportion),
        group_allocation=group_allocation.get_group_allocation_for_target(Target.TREATMENT_PROPORTION),
        search_direction=SearchDirection[search_direction],
        dropout_rate=DropOutRate(dropout_rate),
    )
    model.solve()

    if full_output:
        return model
    return model.treatment_proportion


def solve_for_reference_proportion(
    alpha: float,
    power: float,
    treatment_proportion: float,
    alternative: str,
    test_type: str,
    group_allocation: GroupAllocation,
    search_direction: str,
    dropout_rate: float = 0,
    full_output: bool = False,
):
    """求解对照组的率

    Args:
        alpha (float): 显著性水平
        power (float): 检验效能
        treatment_proportion (float): 试验组的率
        alternative (str): 备择假设类型，可选值: "TWO_SIDED", "ONE_SIDED"
        test_type (str): 检验类型，可选值: "Z_TEST_POOLED", "Z_TEST_UNPOOLED", "Z_TEST_CC_POOLED", "Z_TEST_CC_UNPOOLED"
        group_allocation (GroupAllocation.ForSize, optional): 样本量分配模式
        search_direction (str): 搜索方向，可选值: "LESS", "GREATER"
        dropout_rate (float, optional): 退出率。默认值: 0
        full_output (bool, optional): 是否输出完整结果。默认值: False
    """

    model = TwoProportion.ForReferenceProportion(
        alpha=Alpha(alpha),
        power=Power(power),
        alternative=Alternative[alternative],
        test_type=TestType[test_type],
        treatment_proportion=Proportion(treatment_proportion),
        group_allocation=group_allocation.get_group_allocation_for_target(Target.REFERENCE_PROPORTION),
        search_direction=SearchDirection[search_direction],
        dropout_rate=DropOutRate(dropout_rate),
    )
    model.solve()

    if full_output:
        return model
    return model.reference_proportion
