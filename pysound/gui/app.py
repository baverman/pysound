from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import MISSING, Field, fields, is_dataclass
from typing import Any, cast, get_type_hints

import customtkinter as ctk  # type: ignore[import-untyped]

from .controls import (
    _FIELD_METADATA_KEY,
    FieldConfig,
    FieldInfo,
    Group,
    Knob,
    Radio,
)
from .widgets import (
    Container,
    Control,
    Panel,
    build_container,
    build_control,
    new_row,
    pack,
)


class App[T]:
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
            self._bind_keyboard()

    def _bind_keyboard(self) -> None:
        """Keep the hidden ``keys`` field in sync with the keyboard."""

        if self.window is None:
            return

        from .. import keys

        def refresh(_event: Any) -> None:
            if self._state is not None:
                setattr(self._state, 'keys', keys.keyboard_state())

        self.window.bind('<KeyPress>', refresh)
        self.window.bind('<KeyRelease>', refresh)

    def build(self, parent: Any | None = None) -> App[T]:
        if self.root is not None:
            return self

        _configure_customtkinter()
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
            self._bind_keyboard()
        return self

    def widget(self, path: str) -> Any:
        return self.widgets[path]

    def mainloop(self) -> None:
        self.build()
        if self.window is None:
            raise RuntimeError('app window was not built')
        self.window.mainloop()

    def play(self, generator: Any) -> None:
        """Run a pysound generator while this app owns the main loop."""

        from ..core import play, scream

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
            raise RuntimeError('app window was not built')
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
        row = new_row(parent)
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
                row = new_row(parent)
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
                container = build_container(
                    default.factory or Panel,
                    row,
                    *default.args,
                    **options,
                )
                pack(container.widget, row, padding=0 if anonymous else 4)
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
                container = build_container(Panel, target.content, label=info.label)
                pack(container.widget, target.content)
                self._build_dataclass(field_type, container.content, field_path)
            elif _is_dataclass_type(field_type):
                container = build_container(
                    config.widget or Panel,
                    row,
                    label=info.label,
                )
                pack(container.widget, row)
                self._build_dataclass(field_type, container.content, field_path)
            else:
                self._build_control(row, info)

            if config.nl or (isinstance(default, Group) and default.nl):
                new_line = True

    def _build_control(self, parent: Any, info: FieldInfo) -> None:
        choices = info.config.choices
        convert: Callable[[float], Any]

        if choices is None:
            if info.type is not float:
                raise TypeError(
                    f'{_format_path(info.path)} has unsupported type {info.type!r}; '
                    'only float fields are currently supported'
                )
            convert = float
            factory = info.config.widget or Knob
        else:
            if info.type is not int:
                raise TypeError(
                    f'{_format_path(info.path)} uses choices and must be an int field'
                )
            convert = int
            factory = info.config.widget or Radio

        def changed(value: float) -> None:
            if self._syncing or self._state is None:
                return
            _set_path(self._state, info.path, convert(value))

        control = build_control(factory, parent, info, changed)
        pack(control.widget, parent, expand=False)
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
) -> App[T]:
    """Build an app from a root dataclass type."""

    return App(description, parent, title=title, padding=padding).build()


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


def _configure_customtkinter() -> None:
    zoom = os.environ.get('CTK_ZOOM')
    if zoom is not None:
        ctk.set_widget_scaling(float(zoom))
    ctk.set_appearance_mode(ctk.get_appearance_mode())


def _config_for(model_field: Field[Any]) -> FieldConfig:
    value = model_field.metadata.get(_FIELD_METADATA_KEY)
    return value if isinstance(value, FieldConfig) else FieldConfig()


def _field_default(model_field: Field[Any]) -> Any:
    if model_field.default is not MISSING:
        return model_field.default
    if model_field.default_factory is not MISSING:
        return model_field.default_factory()
    return None


def _is_dataclass_type(value: Any) -> bool:
    return isinstance(value, type) and is_dataclass(value)


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
