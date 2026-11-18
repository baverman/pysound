from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import MISSING, dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import Any, Protocol, overload

_FIELD_METADATA_KEY = 'pysound.gui'


class Transformer(Protocol):
    """Bidirectional mapping between normalized control and value positions."""

    def __call__(self, value: float) -> float: ...

    def inverse(self, value: float) -> float: ...


@dataclass(frozen=True)
class Power:
    exponent: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.exponent) or self.exponent <= 0:
            raise ValueError('exponent must be a positive finite number')

    def __call__(self, value: float) -> float:
        return float(value**self.exponent)

    def inverse(self, value: float) -> float:
        return float(value ** (1 / self.exponent))


pow2 = Power(2)
pow3 = Power(3)


@dataclass(frozen=True)
class InvPower:
    exponent: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.exponent) or self.exponent <= 0:
            raise ValueError('exponent must be a positive finite number')

    def __call__(self, value: float) -> float:
        return float(1 - (1 - value) ** self.exponent)

    def inverse(self, value: float) -> float:
        return float(1 - (1 - value) ** (1 / self.exponent))


invpow2 = InvPower(2)
invpow3 = InvPower(3)


@dataclass(frozen=True)
class Exp:
    """Exponential shape, shared with the envelope curvature.

    k > 0 is a fast start, k < 0 is a slow start, k = 0 is linear.
    """

    k: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.k):
            raise ValueError('k must be finite')

    def __call__(self, value: float) -> float:
        if self.k == 0:
            return float(value)
        return float(math.expm1(-self.k * value) / math.expm1(-self.k))

    def inverse(self, value: float) -> float:
        if self.k == 0:
            return float(value)
        return float(-math.log1p(value * math.expm1(-self.k)) / self.k)


class Scale:
    """Horizontal float slider."""


class VScale:
    """Vertical float slider."""


class Radio:
    """Choice control which stores the index of the selected label."""


class Knob:
    """Rotating float control."""


@dataclass(frozen=True)
class Group:
    factory: Any | None
    args: tuple[Any, ...]
    options: Mapping[str, Any]
    nl: bool = False


@dataclass(frozen=True)
class FieldConfig:
    min: float | None = None
    max: float | None = None
    map: Transformer | None = None
    resolution: float | None = None
    label: str | None = None
    hidden: bool = False
    widget: Any = None
    group: Group | None = None
    nl: bool = False
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class FieldInfo:
    name: str
    path: tuple[str, ...]
    type: Any
    default: Any
    config: FieldConfig

    @property
    def label(self) -> str:
        return self.config.label or self.name.replace('_', ' ').title()


def group(factory: Any = None, *args: Any, nl: bool = False, **options: Any) -> Group:
    """Declare an ordered layout field which receives other fields."""

    return Group(factory, args, MappingProxyType(dict(options)), nl)


@overload
def field[T](
    value: T,
    *,
    range: tuple[float, float] | None = None,
    map: Transformer | None = None,
    resolution: float | None = None,
    choices: tuple[str, ...] | None = None,
    label: str | None = None,
    hidden: bool = False,
    widget: Any = None,
    group: Group | None = None,
    nl: bool = False,
) -> T: ...


@overload
def field(
    *,
    range: tuple[float, float] | None = None,
    map: Transformer | None = None,
    resolution: float | None = None,
    choices: tuple[str, ...] | None = None,
    label: str | None = None,
    hidden: bool = False,
    widget: Any = None,
    group: Group | None = None,
    nl: bool = False,
) -> Any: ...


def field(
    value: Any = MISSING,
    *,
    range: tuple[float, float] | None = None,
    map: Transformer | None = None,
    resolution: float | None = None,
    choices: tuple[str, ...] | None = None,
    label: str | None = None,
    hidden: bool = False,
    widget: Any = None,
    group: Group | None = None,
    nl: bool = False,
) -> Any:
    """Declare a dataclass field and its UI configuration."""

    minimum, maximum = range if range is not None else (None, None)
    if minimum is not None and minimum == maximum:
        raise ValueError('range values must be different')
    if resolution is not None and resolution <= 0:
        raise ValueError('resolution must be positive')
    if choices is not None and not choices:
        raise ValueError('choices must not be empty')

    config = FieldConfig(
        min=minimum,
        max=maximum,
        map=map,
        resolution=resolution,
        choices=choices,
        label=label,
        hidden=hidden,
        widget=widget,
        group=group,
        nl=nl,
    )
    metadata = {_FIELD_METADATA_KEY: config}
    if value is MISSING:
        return dataclass_field(metadata=metadata)
    return dataclass_field(default=value, metadata=metadata)


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _effective_resolution(config: FieldConfig) -> float | None:
    if config.resolution is not None:
        return config.resolution
    if config.map is not None:
        return 1 / 200
    return None


def _range_for(info: FieldInfo) -> tuple[float, float]:
    default = float(info.default) if info.default is not None else 0.0
    minimum = info.config.min if info.config.min is not None else min(0.0, default)
    maximum = info.config.max if info.config.max is not None else max(1.0, default)
    if minimum == maximum:
        maximum = minimum + 1.0
    return float(minimum), float(maximum)


def _control_range(info: FieldInfo) -> tuple[float, float]:
    if info.config.map is not None:
        return 0.0, 1.0
    return _range_for(info)


def _value_from_control(
    control: float,
    minimum: float,
    maximum: float,
    transformer: Transformer | None,
) -> float:
    if transformer is None:
        return control
    normalized = _clamp_unit(float(transformer(_clamp_unit(control))))
    return minimum + normalized * (maximum - minimum)


def _control_from_value(
    value: float,
    minimum: float,
    maximum: float,
    transformer: Transformer | None,
) -> float:
    if transformer is None:
        return value
    normalized = _clamp_unit((value - minimum) / (maximum - minimum))
    return _clamp_unit(float(transformer.inverse(normalized)))
