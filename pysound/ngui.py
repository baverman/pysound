from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import MISSING, Field, dataclass, fields, is_dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import Any, Protocol, cast, get_type_hints, overload

import customtkinter as ctk  # type: ignore[import-untyped]

ctk.set_appearance_mode(ctk.get_appearance_mode())

ChangeCallback = Callable[[float], None]
_FIELD_METADATA_KEY = 'pysound.ngui'


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


@dataclass
class Control:
    widget: Any
    set_value: Callable[[float], None]


@dataclass
class Container:
    widget: Any
    content: Any


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _effective_resolution(config: FieldConfig) -> float | None:
    if config.resolution is not None:
        return config.resolution
    if config.map is not None:
        return 1 / 200
    return None


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


class Scale:
    """Default float-field widget."""

    @classmethod
    def build(cls, parent: Any, info: FieldInfo, command: ChangeCallback) -> Control:
        config = info.config
        minimum, maximum = _range_for(info)
        control_minimum, control_maximum = _control_range(info)
        resolution = _effective_resolution(config)
        frame = ctk.CTkFrame(parent, fg_color='transparent')
        label = ctk.CTkLabel(frame, text=info.label, anchor='w')
        label.pack(fill='x')

        steps = None
        if resolution is not None:
            steps = max(1, round(abs(control_maximum - control_minimum) / resolution))

        def changed(control: float) -> None:
            command(_value_from_control(control, minimum, maximum, config.map))

        slider = ctk.CTkSlider(
            frame,
            from_=control_minimum,
            to=control_maximum,
            number_of_steps=steps,
            command=changed,
        )
        slider.pack(fill='x', expand=True)

        def set_value(value: float) -> None:
            slider.set(_control_from_value(value, minimum, maximum, config.map))

        return Control(frame, set_value)


class Knob:
    """Rotating float control drawn on a customtkinter canvas."""

    @classmethod
    def build(cls, parent: Any, info: FieldInfo, command: ChangeCallback) -> Control:
        knob = _Knob(parent, info, command)
        return Control(knob.frame, knob.set)


class _Knob:
    def __init__(
        self,
        parent: Any,
        info: FieldInfo,
        command: ChangeCallback,
        size: int = 88,
    ) -> None:
        self.minimum, self.maximum = _range_for(info)
        self.map = info.config.map
        self.control_minimum, self.control_maximum = _control_range(info)
        self.resolution = _effective_resolution(info.config)
        self.command = command
        self.size = size
        self.value = self.minimum
        self._drag_y = 0
        self._drag_control = self.control_minimum

        self.frame = ctk.CTkFrame(parent, fg_color='transparent')
        ctk.CTkLabel(self.frame, text=info.label).pack(fill='x')
        self.canvas = ctk.CTkCanvas(
            self.frame,
            width=size,
            height=size,
            highlightthickness=0,
            background=_widget_background(parent),
        )
        self.canvas.pack()

        self.canvas.bind('<ButtonPress-1>', self._press)
        self.canvas.bind('<B1-Motion>', self._drag)
        self.canvas.bind('<MouseWheel>', self._wheel)
        self.canvas.bind('<Button-4>', self._wheel)
        self.canvas.bind('<Button-5>', self._wheel)
        self._draw()

    def set(self, value: float) -> None:
        self.value = self._clamp_value(float(value))
        self._draw()

    def _set_from_user(self, control: float) -> None:
        control = self._quantize(self._clamp_control(control))
        value = _value_from_control(control, self.minimum, self.maximum, self.map)
        self.set(value)
        self.command(value)

    def _press(self, event: Any) -> None:
        self._drag_y = event.y_root
        self._drag_control = _control_from_value(self.value, self.minimum, self.maximum, self.map)

    def _drag(self, event: Any) -> None:
        distance = self._drag_y - event.y_root
        span = self.control_maximum - self.control_minimum
        self._set_from_user(self._drag_control + distance * span / 150)

    def _wheel(self, event: Any) -> None:
        direction = 1 if getattr(event, 'delta', 0) > 0 or getattr(event, 'num', 0) == 4 else -1
        span = self.control_maximum - self.control_minimum
        step = self.resolution or span / 100
        control = _control_from_value(self.value, self.minimum, self.maximum, self.map)
        self._set_from_user(control + direction * step)

    def _clamp_value(self, value: float) -> float:
        low, high = sorted((self.minimum, self.maximum))
        return min(high, max(low, value))

    def _clamp_control(self, value: float) -> float:
        low, high = sorted((self.control_minimum, self.control_maximum))
        return min(high, max(low, value))

    def _quantize(self, value: float) -> float:
        if self.resolution is None:
            return value
        steps = round((value - self.control_minimum) / self.resolution)
        return self.control_minimum + steps * self.resolution

    def _draw(self) -> None:
        control = _control_from_value(self.value, self.minimum, self.maximum, self.map)
        normalized = (control - self.control_minimum) / (
            self.control_maximum - self.control_minimum
        )
        normalized = _clamp_unit(normalized)
        padding = 9
        bounds = (padding, padding, self.size - padding, self.size - padding)
        track = _appearance_color(ctk.ThemeManager.theme['CTkProgressBar']['fg_color'])
        accent = _appearance_color(ctk.ThemeManager.theme['CTkProgressBar']['progress_color'])
        indicator = _appearance_color(ctk.ThemeManager.theme['CTkLabel']['text_color'])

        self.canvas.delete('all')
        self.canvas.create_arc(
            *bounds,
            start=225,
            extent=-270,
            style='arc',
            outline=track,
            width=7,
        )
        self.canvas.create_arc(
            *bounds,
            start=225,
            extent=-270 * normalized,
            style='arc',
            outline=accent,
            width=7,
        )

        center = self.size / 2
        radius = self.size / 2 - 18
        angle = math.radians(225 - 270 * normalized)
        end_x = center + math.cos(angle) * radius
        end_y = center - math.sin(angle) * radius
        self.canvas.create_oval(
            center - 4,
            center - 4,
            center + 4,
            center + 4,
            fill=indicator,
            outline='',
        )
        self.canvas.create_line(
            center,
            center,
            end_x,
            end_y,
            fill=indicator,
            width=3,
            capstyle='round',
        )
        self.canvas.create_text(
            center,
            self.size - 10,
            text=self._format_value(),
            fill=indicator,
            font='TkDefaultFont',
        )

    def _format_value(self) -> str:
        if self.map is not None or self.resolution is None or self.resolution >= 1:
            return f'{self.value:g}'
        precision = max(0, math.ceil(-math.log10(self.resolution)))
        return f'{self.value:.{precision}f}'


class Panel:
    """Default container for nested dataclasses and groups."""

    @classmethod
    def build(
        cls,
        parent: Any,
        *,
        label: str | None = None,
        orientation: str = 'horizontal',
        padding: int = 8,
        gap: int = 12,
        border_width: int = 1,
        **kwargs: Any,
    ) -> Container:
        frame = ctk.CTkFrame(parent, fg_color='transparent')
        body = ctk.CTkFrame(frame, border_width=border_width, **kwargs)
        title_height = 18
        body.pack(
            fill='both',
            expand=True,
            pady=(title_height // 2 if label else 0, 0),
        )
        if label:
            ctk.CTkLabel(
                frame,
                text=f' {label} ',
                height=title_height,
                anchor='w',
                fg_color=body.cget('fg_color'),
            ).place(x=padding, y=0)
        content = ctk.CTkFrame(body, fg_color='transparent')
        content.pack(fill='both', expand=True, padx=padding, pady=padding)
        content._ngui_orientation = orientation
        content._ngui_gap = gap
        return Container(frame, content)


@dataclass(frozen=True)
class Group:
    factory: Any
    args: tuple[Any, ...]
    options: Mapping[str, Any]
    nl: bool = False


def group(factory: Any = Panel, *args: Any, nl: bool = False, **options: Any) -> Group:
    """Declare an ordered layout field which receives other fields."""

    return Group(factory, args, MappingProxyType(dict(options)), nl)


@overload
def field[T](
    value: T,
    *,
    range: tuple[float, float] | None = None,
    map: Transformer | None = None,
    resolution: float | None = None,
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

    config = FieldConfig(
        min=minimum,
        max=maximum,
        map=map,
        resolution=resolution,
        label=label,
        hidden=hidden,
        widget=widget,
        group=group,
        nl=nl,
    )
    metadata = {_FIELD_METADATA_KEY: config}
    if value is MISSING:
        result = dataclass_field(metadata=metadata)
    else:
        result = dataclass_field(default=value, metadata=metadata)
    return result


class GUI[T]:
    """Build and bind a customtkinter tree for a dataclass type."""

    def __init__(
        self,
        description: type[T],
        parent: Any | None = None,
        *,
        title: str | None = None,
        padding: int = 8,
    ) -> None:
        if not isinstance(description, type) or not is_dataclass(description):
            raise TypeError('description must be a dataclass type')
        self.description = description
        self.parent = parent
        self.title = title or description.__name__
        self.padding = padding
        self._background: Any | None = None
        self.root: Any | None = None
        self.window: Any | None = None
        self.widgets: dict[str, Any] = {}
        self._controls: dict[tuple[str, ...], Control] = {}
        self._groups: dict[int, Container] = {}
        self._state: T | None = None
        self._syncing = False

    @property
    def state(self) -> T | None:
        return self._state

    def bind(self, state: T) -> None:
        if not isinstance(state, self.description):
            raise TypeError(f'state must be an instance of {self.description.__name__}')
        self._state = state
        if self.root is not None:
            self._load_state()

    def build(self, parent: Any | None = None) -> GUI[T]:
        if self.root is not None:
            return self

        parent = parent if parent is not None else self.parent
        if parent is None:
            self.window = ctk.CTk()
            self.window.title(self.title)
            parent = self.window
        else:
            self.window = parent.winfo_toplevel()

        self._background = ctk.CTkFrame(parent)
        self._background.pack(fill='both', expand=True)
        self.root = ctk.CTkFrame(self._background, fg_color='transparent')
        self.root.pack(fill='both', expand=True, padx=self.padding, pady=self.padding)
        self._build_dataclass(self.description, self.root, ())
        if self._state is not None:
            self._load_state()
        return self

    def widget(self, path: str) -> Any:
        return self.widgets[path]

    def mainloop(self) -> None:
        self.build()
        if self.window is None:
            raise RuntimeError('GUI window was not built')
        self.window.mainloop()

    def play(self, generator: Any) -> None:
        """Run a pysound generator while this GUI owns the main loop."""

        from .core import play, scream

        self.build()
        if self._state is None:
            raise RuntimeError('no state is bound')
        stop = threading.Event()
        worker = threading.Thread(
            target=scream(play),
            args=(self._state, generator),
            kwargs={'stop': stop},
            daemon=True,
        )
        worker.start()
        if self.window is None:
            raise RuntimeError('GUI window was not built')
        try:
            self.window.mainloop()
        finally:
            stop.set()
            worker.join()

    def _build_dataclass(
        self,
        model_type: type[Any],
        parent: Any,
        path: tuple[str, ...],
    ) -> None:
        groups: dict[int, Container] = {}
        hints = get_type_hints(model_type)
        row = _new_row(parent)
        new_line = False

        for model_field in fields(model_type):
            config = _config_for(model_field)
            if config.hidden:
                continue

            field_type = hints.get(model_field.name, model_field.type)
            field_path = (*path, model_field.name)
            default = _field_default(model_field)
            info = FieldInfo(model_field.name, field_path, field_type, default, config)

            if config.nl and config.group is not None:
                raise ValueError(f'{_format_path(field_path)} cannot use nl and group together')
            if new_line and config.group is None:
                row = _new_row(parent)
                new_line = False

            if isinstance(default, Group):
                anonymous = model_field.name.startswith('_')
                options = dict(default.options)
                if anonymous:
                    options['label'] = None
                    options['padding'] = 0
                    options['border_width'] = 0
                else:
                    options.setdefault('label', info.label)
                container = _build_container(
                    default.factory,
                    row,
                    *default.args,
                    **options,
                )
                _pack(container.widget, row, padding=0 if anonymous else 4)
                groups[id(default)] = container
                self._groups[id(default)] = container
                self.widgets[_format_path(field_path)] = container.widget
            elif config.group is not None:
                target = groups.get(id(config.group))
                if target is None:
                    raise ValueError(
                        f'{_format_path(field_path)} refers to a group which must be '
                        'declared earlier in the same dataclass'
                    )
                if not _is_dataclass_type(field_type):
                    raise TypeError(
                        f'{_format_path(field_path)} must be a dataclass to use a group'
                    )
                container = _build_container(Panel, target.content, label=info.label)
                _pack(container.widget, target.content)
                self._build_dataclass(field_type, container.content, field_path)
            elif _is_dataclass_type(field_type):
                container = _build_container(
                    config.widget or Panel,
                    row,
                    label=info.label,
                )
                _pack(container.widget, row)
                self._build_dataclass(field_type, container.content, field_path)
            else:
                self._build_control(row, info)

            if config.nl or (isinstance(default, Group) and default.nl):
                new_line = True

    def _build_control(self, parent: Any, info: FieldInfo) -> None:
        if info.type is not float:
            raise TypeError(
                f'{_format_path(info.path)} has unsupported type {info.type!r}; '
                'only float fields are currently supported'
            )

        def changed(value: float) -> None:
            if self._syncing or self._state is None:
                return
            _set_path(self._state, info.path, float(value))

        control = _build_control(info.config.widget or Knob, parent, info, changed)
        _pack(control.widget, parent, expand=False)
        self._controls[info.path] = control
        self.widgets[_format_path(info.path)] = control.widget

    def _load_state(self) -> None:
        if self._state is None:
            return
        self._syncing = True
        try:
            for path, control in self._controls.items():
                control.set_value(float(_get_path(self._state, path)))
        finally:
            self._syncing = False


def build[T](
    description: type[T],
    parent: Any | None = None,
    *,
    title: str | None = None,
    padding: int = 8,
) -> GUI[T]:
    """Build a GUI from a root dataclass type."""

    return GUI(description, parent, title=title, padding=padding).build()


def make[T](description: type[T]) -> T:
    """Recursively construct a dataclass state from annotations and defaults."""

    if not _is_dataclass_type(description):
        raise TypeError('description must be a dataclass type')

    hints = get_type_hints(description)
    values: dict[str, Any] = {}
    for model_field in fields(cast(Any, description)):
        if model_field.default is not MISSING or model_field.default_factory is not MISSING:
            continue
        field_type = hints.get(model_field.name, model_field.type)
        if not _is_dataclass_type(field_type):
            raise TypeError(
                f'{description.__name__}.{model_field.name} has no default and cannot be made'
            )
        values[model_field.name] = make(field_type)
    return description(**values)


def _config_for(model_field: Field[Any]) -> FieldConfig:
    value = model_field.metadata.get(_FIELD_METADATA_KEY)
    return value if isinstance(value, FieldConfig) else FieldConfig()


def _field_default(model_field: Field[Any]) -> Any:
    if model_field.default is not MISSING:
        return model_field.default
    if model_field.default_factory is not MISSING:
        return model_field.default_factory()
    return None


def _appearance_color(color: Any) -> str:
    if isinstance(color, (tuple, list)):
        index = 0 if ctk.get_appearance_mode() == 'Light' else 1
        return str(color[index])
    return str(color)


def _widget_background(widget: Any) -> str:
    current = widget
    while current is not None:
        try:
            color = current.cget('fg_color')
        except (AttributeError, ValueError):
            color = None
        if color not in (None, 'transparent'):
            return _appearance_color(color)
        current = getattr(current, 'master', None)
    return _appearance_color(ctk.ThemeManager.theme['CTk']['fg_color'])


def _range_for(info: FieldInfo) -> tuple[float, float]:
    default = float(info.default) if info.default is not None else 0.0
    minimum = info.config.min if info.config.min is not None else min(0.0, default)
    maximum = info.config.max if info.config.max is not None else max(1.0, default)
    if minimum == maximum:
        maximum = minimum + 1.0
    return float(minimum), float(maximum)


def _is_dataclass_type(value: Any) -> bool:
    return isinstance(value, type) and is_dataclass(value)


def _build_control(
    factory: Any,
    parent: Any,
    info: FieldInfo,
    command: ChangeCallback,
) -> Control:
    builder = getattr(factory, 'build', None)
    result = builder(parent, info, command) if builder else factory(parent, info, command)
    if isinstance(result, Control):
        return result
    if hasattr(result, 'set'):
        return Control(result, result.set)
    raise TypeError('field widget factory must return Control or an object with set()')


def _build_container(
    factory: Any,
    parent: Any,
    *args: Any,
    **options: Any,
) -> Container:
    builder = getattr(factory, 'build', None)
    result = builder(parent, *args, **options) if builder else factory(parent, *args, **options)
    if isinstance(result, Container):
        return result
    return Container(result, result)


def _new_row(parent: Any) -> Any:
    row = ctk.CTkFrame(parent, fg_color='transparent')
    row._ngui_orientation = 'horizontal'
    row.pack(fill='x')
    return row


def _pack(
    widget: Any,
    parent: Any,
    *,
    padding: int = 4,
    expand: bool = True,
) -> None:
    orientation = getattr(parent, '_ngui_orientation', 'vertical')
    if orientation == 'horizontal':
        if expand:
            widget.pack(side='left', fill='both', expand=True, padx=padding, pady=padding)
        else:
            widget.pack(side='left', padx=padding, pady=padding)
    else:
        gap = getattr(parent, '_ngui_gap', None)
        if gap is None:
            widget.pack(fill='x', padx=padding, pady=padding)
        else:
            pady = (gap, 0) if parent.pack_slaves() else 0
            widget.pack(fill='x', padx=padding, pady=pady)


def _get_path(state: Any, path: tuple[str, ...]) -> Any:
    value = state
    for name in path:
        value = getattr(value, name)
    return value


def _set_path(state: Any, path: tuple[str, ...], value: Any) -> None:
    target = state
    for name in path[:-1]:
        target = getattr(target, name)
    setattr(target, path[-1], value)


def _format_path(path: tuple[str, ...]) -> str:
    return '.'.join(path)


__all__ = [
    'GUI',
    'Container',
    'Control',
    'FieldConfig',
    'FieldInfo',
    'Group',
    'Knob',
    'Panel',
    'Scale',
    'build',
    'field',
    'group',
    'make',
]
