from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Mapping
from dataclasses import MISSING, Field, dataclass, fields, is_dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import Any, get_type_hints

import customtkinter as ctk

ChangeCallback = Callable[[float], None]
_FIELD_METADATA_KEY = 'pysound.ngui'


@dataclass(frozen=True)
class FieldConfig:
    min: float | None = None
    max: float | None = None
    resolution: float | None = None
    label: str | None = None
    hidden: bool = False
    widget: Any = None
    group: Group | None = None


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


class Scale:
    """Default float-field widget."""

    @classmethod
    def build(cls, parent: Any, info: FieldInfo, command: ChangeCallback) -> Control:
        config = info.config
        minimum, maximum = _range_for(info)
        frame = ctk.CTkFrame(parent, fg_color='transparent')
        label = ctk.CTkLabel(frame, text=info.label, anchor='w')
        label.pack(fill='x')

        steps = None
        if config.resolution is not None:
            if config.resolution <= 0:
                raise ValueError(f'{_format_path(info.path)}: resolution must be positive')
            steps = max(1, round(abs(maximum - minimum) / config.resolution))

        slider = ctk.CTkSlider(
            frame,
            from_=minimum,
            to=maximum,
            number_of_steps=steps,
            command=command,
        )
        slider.pack(fill='x', expand=True)
        return Control(frame, slider.set)


class Panel:
    """Default container for nested dataclasses and groups."""

    @classmethod
    def build(
        cls,
        parent: Any,
        *,
        label: str | None = None,
        orientation: str = 'vertical',
        padding: int = 8,
        **kwargs: Any,
    ) -> Container:
        frame = ctk.CTkFrame(parent, **kwargs)
        if label:
            ctk.CTkLabel(frame, text=label, anchor='w').pack(
                fill='x', padx=padding, pady=(padding, 0)
            )
        content = ctk.CTkFrame(frame, fg_color='transparent')
        content.pack(fill='both', expand=True, padx=padding, pady=padding)
        content._ngui_orientation = orientation
        return Container(frame, content)


@dataclass(frozen=True)
class Group:
    factory: Any
    args: tuple[Any, ...]
    options: Mapping[str, Any]


def group(factory: Any = Panel, *args: Any, **options: Any) -> Group:
    """Declare an ordered layout field which receives other fields."""

    return Group(factory, args, MappingProxyType(dict(options)))


def field[T](
    value: T = MISSING,
    min: float | None = None,
    max: float | None = None,
    resolution: float | None = None,
    label: str | None = None,
    hidden: bool = False,
    widget: Any = None,
    group: Group | None = None,
    *,
    default_factory: Callable[[], T] | Any = MISSING,
) -> T:
    """Declare a dataclass field and its UI configuration."""

    if value is not MISSING and default_factory is not MISSING:
        raise TypeError('value and default_factory are mutually exclusive')
    if min is not None and max is not None and min == max:
        raise ValueError('min and max must be different')

    config = FieldConfig(min, max, resolution, label, hidden, widget, group)
    metadata = {_FIELD_METADATA_KEY: config}

    if default_factory is not MISSING:
        result = dataclass_field(default_factory=default_factory, metadata=metadata)
    elif value is MISSING:
        result = dataclass_field(metadata=metadata)
    elif is_dataclass(value) and not isinstance(value, type):
        # Dataclasses reject mutable dataclass defaults. Preserve the concise
        # field(State(), ...) form by creating an independent copy per instance.
        result = dataclass_field(
            default_factory=lambda template=value: copy.deepcopy(template),
            metadata=metadata,
        )
    else:
        result = dataclass_field(default=value, metadata=metadata)
    return result  # type: ignore[return-value]


class GUI[T]:
    """Build and bind a customtkinter tree for a dataclass type."""

    def __init__(
        self,
        description: type[T],
        parent: Any | None = None,
        *,
        title: str | None = None,
    ) -> None:
        if not isinstance(description, type) or not is_dataclass(description):
            raise TypeError('description must be a dataclass type')
        self.description = description
        self.parent = parent
        self.title = title or description.__name__
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

        self.root = ctk.CTkFrame(parent)
        self.root.pack(fill='both', expand=True)
        self._build_dataclass(self.description, self.root, ())
        if self._state is not None:
            self._load_state()
        return self

    def widget(self, path: str) -> Any:
        return self.widgets[path]

    def mainloop(self) -> None:
        self.build()
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

        for model_field in fields(model_type):
            config = _config_for(model_field)
            if config.hidden:
                continue

            field_type = hints.get(model_field.name, model_field.type)
            field_path = (*path, model_field.name)
            default = _field_default(model_field)
            info = FieldInfo(model_field.name, field_path, field_type, default, config)

            if isinstance(default, Group):
                anonymous = model_field.name.startswith('_')
                options = dict(default.options)
                if anonymous:
                    options['label'] = None
                    options['padding'] = 0
                else:
                    options.setdefault('label', info.label)
                container = _build_container(
                    default.factory,
                    parent,
                    *default.args,
                    **options,
                )
                _pack(container.widget, parent, padding=0 if anonymous else 4)
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
                    parent,
                    label=info.label,
                )
                _pack(container.widget, parent)
                self._build_dataclass(field_type, container.content, field_path)
            else:
                self._build_control(parent, info)

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

        control = _build_control(info.config.widget or Scale, parent, info, changed)
        _pack(control.widget, parent)
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
) -> GUI[T]:
    """Build a GUI from a root dataclass type."""

    return GUI(description, parent, title=title).build()


def _config_for(model_field: Field[Any]) -> FieldConfig:
    value = model_field.metadata.get(_FIELD_METADATA_KEY)
    return value if isinstance(value, FieldConfig) else FieldConfig()


def _field_default(model_field: Field[Any]) -> Any:
    if model_field.default is not MISSING:
        return model_field.default
    if model_field.default_factory is not MISSING:
        return model_field.default_factory()
    return None


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


def _pack(widget: Any, parent: Any, *, padding: int = 4) -> None:
    orientation = getattr(parent, '_ngui_orientation', 'vertical')
    if orientation == 'horizontal':
        widget.pack(side='left', fill='both', expand=True, padx=padding, pady=padding)
    else:
        widget.pack(fill='x', padx=padding, pady=padding)


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
    'Panel',
    'Scale',
    'build',
    'field',
    'group',
]
