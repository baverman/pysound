from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import customtkinter as ctk  # type: ignore[import-untyped]
from PIL import Image, ImageDraw, ImageTk

from .controls import (
    FieldInfo,
    Knob,
    Radio,
    Scale,
    VScale,
    _clamp_unit,
    _control_from_value,
    _control_range,
    _effective_resolution,
    _range_for,
    _value_from_control,
)

ChangeCallback = Callable[[float], None]
_FRAME_INTERVAL = 1 / 60


@dataclass
class Control:
    widget: Any
    set_value: Callable[[float], None]


@dataclass
class Container:
    widget: Any
    content: Any


def _build_slider(
    parent: Any,
    info: FieldInfo,
    command: ChangeCallback,
    *,
    orientation: str,
) -> Control:
    config = info.config
    minimum, maximum = _range_for(info)
    control_minimum, control_maximum = _control_range(info)
    resolution = _effective_resolution(config)
    frame = ctk.CTkFrame(parent, fg_color='transparent')
    label = ctk.CTkLabel(frame, text=info.label)
    label.pack(fill='x')

    steps = None
    if resolution is not None:
        steps = max(1, round(abs(control_maximum - control_minimum) / resolution))

    def changed(control: float) -> None:
        command(_value_from_control(control, minimum, maximum, config.map))

    options: dict[str, Any] = {}
    if orientation == 'vertical':
        options['height'] = 120

    slider = ctk.CTkSlider(
        frame,
        orientation=orientation,
        from_=control_minimum,
        to=control_maximum,
        number_of_steps=steps,
        command=changed,
        **options,
    )
    slider.pack(fill='x' if orientation == 'horizontal' else 'y', expand=True)

    def set_value(value: float) -> None:
        slider.set(_control_from_value(value, minimum, maximum, config.map))

    return Control(frame, set_value)


def _build_radio(parent: Any, info: FieldInfo, command: ChangeCallback) -> Control:
    labels = info.config.choices
    if not labels:
        raise ValueError(f'{_format_path(info.path)}: choices must not be empty')

    frame = ctk.CTkFrame(parent, fg_color='transparent')
    ctk.CTkLabel(frame, text=info.label).pack(fill='x')

    def changed(label: str) -> None:
        command(float(labels.index(label)))

    buttons = ctk.CTkSegmentedButton(
        frame, values=list(labels), orientation='vertical', command=changed
    )
    buttons.pack(fill='y', expand=True)

    def set_value(value: float) -> None:
        buttons.set(labels[int(value)])

    return Control(frame, set_value)


def _build_knob(parent: Any, info: FieldInfo, command: ChangeCallback) -> Control:
    knob = _Knob(parent, info, command)
    return Control(knob.frame, knob.set)


class _Knob:
    def __init__(
        self,
        parent: Any,
        info: FieldInfo,
        command: ChangeCallback,
        size: int = 56,
        value_gap: int = 8,
    ) -> None:
        self.minimum, self.maximum = _range_for(info)
        self.map = info.config.map
        self.control_minimum, self.control_maximum = _control_range(info)
        self.resolution = _effective_resolution(info.config)
        self.command = command
        self.size = size
        self.value_gap = value_gap
        self.value = self.minimum
        self._drag_y = 0
        self._drag_control = self.control_minimum

        self.frame = ctk.CTkFrame(parent, fg_color='transparent')
        ctk.CTkLabel(self.frame, text=info.label).pack(fill='x')
        self.scaling = ctk.ScalingTracker.get_widget_scaling(self.frame)
        canvas_size = round(size * self.scaling)
        value_gap_size = round(value_gap * self.scaling)
        self.canvas = ctk.CTkCanvas(
            self.frame,
            width=canvas_size,
            height=canvas_size + value_gap_size,
            highlightthickness=0,
            background=_widget_background(parent),
        )
        self.canvas.pack()

        supersampling = 4
        self._canvas_size = (canvas_size, canvas_size + value_gap_size)
        self._render_size = tuple(value * supersampling for value in self._canvas_size)
        background = _image_color(self.canvas, str(self.canvas.cget('background')))
        self._render_image = Image.new('RGB', self._render_size, background)
        self._image_draw = ImageDraw.Draw(self._render_image)
        display_image = Image.new('RGB', self._canvas_size, background)
        self._knob_image = ImageTk.PhotoImage(display_image, master=self.canvas)
        self._image_item = self.canvas.create_image(
            0, 0, anchor='nw', image=self._knob_image
        )
        indicator = _appearance_color(ctk.ThemeManager.theme['CTkLabel']['text_color'])
        font = ctk.ThemeManager.theme['CTkFont']
        self._text_item = self.canvas.create_text(
            canvas_size / 2,
            canvas_size + value_gap_size - round(10 * self.scaling),
            text=self._format_value(),
            fill=indicator,
            font=(font['family'], -round(font['size'] * self.scaling)),
        )

        self.canvas.bind('<ButtonPress-1>', self._press)
        self.canvas.bind('<B1-Motion>', self._drag)
        self.canvas.bind('<MouseWheel>', self._wheel)
        self.canvas.bind('<Button-4>', self._wheel)
        self.canvas.bind('<Button-5>', self._wheel)
        self._redraw_job: str | None = None
        self._last_draw = 0.0
        self._draw()
        self._last_draw = time.monotonic()

    def set(self, value: float) -> None:
        value = self._clamp_value(float(value))
        if value == self.value:
            return
        self.value = value
        self._request_draw()

    def _request_draw(self) -> None:
        if self._redraw_job is not None:
            return
        elapsed = time.monotonic() - self._last_draw
        remaining = max(0.0, _FRAME_INTERVAL - elapsed)
        delay = math.ceil(remaining * 1000)
        self._redraw_job = self.canvas.after(delay, self._flush_draw)

    def _flush_draw(self) -> None:
        self._redraw_job = None
        self._last_draw = time.monotonic()
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
        self._set_from_user(self._drag_control + distance * span / (150 * self.scaling))

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

    def _normalized_value(self) -> float:
        control = _control_from_value(self.value, self.minimum, self.maximum, self.map)
        normalized = (control - self.control_minimum) / (
            self.control_maximum - self.control_minimum
        )
        return _clamp_unit(normalized)

    def _prepare_image(self) -> Image.Image:
        normalized = self._normalized_value()
        render_width, render_height = self._render_size
        def px(value: float) -> int:
            return round(value * render_width)

        padding = max(1, px(0.02))
        arc_width = max(1, px(0.08))
        bounds = (
            padding,
            padding,
            render_width - 1 - padding,
            render_width - 1 - padding,
        )
        track = _appearance_color(ctk.ThemeManager.theme['CTkProgressBar']['fg_color'])
        accent = _appearance_color(ctk.ThemeManager.theme['CTkProgressBar']['progress_color'])
        indicator = _appearance_color(ctk.ThemeManager.theme['CTkLabel']['text_color'])
        track_image = _image_color(self.canvas, track)
        accent_image = _image_color(self.canvas, accent)
        indicator_image = _image_color(self.canvas, indicator)
        background_image = _image_color(self.canvas, str(self.canvas.cget('background')))

        draw = self._image_draw
        draw.rectangle(
            (0, 0, render_width - 1, render_height - 1),
            fill=background_image,
        )
        draw.arc(
            bounds,
            start=135,
            end=405,
            fill=track_image,
            width=arc_width,
        )
        if normalized > 0:
            draw.arc(
                bounds,
                start=135,
                end=135 + 270 * normalized,
                fill=accent_image,
                width=arc_width,
            )

        center = px(0.5)
        angle = math.radians(225 - 270 * normalized)
        end_x = px(0.5 + math.cos(angle) * 0.30)
        end_y = px(0.5 - math.sin(angle) * 0.30)
        line_width = max(1, px(0.05))
        line_points = [(center, center), (end_x, end_y)]
        draw.line(line_points, fill=indicator_image, width=line_width)

        cap_radius = line_width / 2
        draw.ellipse(
            (
                round(end_x - cap_radius),
                round(end_y - cap_radius),
                round(end_x + cap_radius),
                round(end_y + cap_radius),
            ),
            fill=indicator_image,
        )

        center_radius = px(0.05)
        draw.ellipse(
            (
                center - center_radius,
                center - center_radius,
                center + center_radius,
                center + center_radius,
            ),
            fill=indicator_image,
        )

        return self._render_image.resize(
            self._canvas_size,
            Image.Resampling.LANCZOS,
        )

    def _draw(self) -> None:
        self._knob_image.paste(self._prepare_image())
        indicator = _appearance_color(ctk.ThemeManager.theme['CTkLabel']['text_color'])
        self.canvas.itemconfigure(
            self._text_item,
            text=self._format_value(),
            fill=indicator,
        )

    def _format_value(self) -> str:
        if abs(self.value) > 9999:
            return str(round(self.value))
        return f'{self.value:.4g}'


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
        content._gui_orientation = orientation
        content._gui_gap = gap
        return Container(frame, content)


def build_control(
    factory: Any,
    parent: Any,
    info: FieldInfo,
    command: ChangeCallback,
) -> Control:
    if factory is Scale:
        return _build_slider(parent, info, command, orientation='horizontal')
    if factory is VScale:
        return _build_slider(parent, info, command, orientation='vertical')
    if factory is Radio:
        return _build_radio(parent, info, command)
    if factory is Knob:
        return _build_knob(parent, info, command)

    builder = getattr(factory, 'build', None)
    result = builder(parent, info, command) if builder else factory(parent, info, command)
    if isinstance(result, Control):
        return result
    if hasattr(result, 'set'):
        return Control(result, result.set)
    raise TypeError('field widget factory must return Control or an object with set()')


def build_container(
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


def new_row(parent: Any) -> Any:
    row = ctk.CTkFrame(parent, fg_color='transparent')
    row._gui_orientation = 'horizontal'
    row.pack(fill='x')
    return row


def pack(
    widget: Any,
    parent: Any,
    *,
    padding: int = 4,
    expand: bool = True,
) -> None:
    orientation = getattr(parent, '_gui_orientation', 'vertical')
    if orientation == 'horizontal':
        if expand:
            widget.pack(
                side='left',
                fill='both',
                expand=True,
                anchor='n',
                padx=padding,
                pady=padding,
            )
        else:
            widget.pack(side='left', anchor='n', padx=padding, pady=padding)
    else:
        gap = getattr(parent, '_gui_gap', None)
        if gap is None:
            widget.pack(fill='x', padx=padding, pady=padding)
        else:
            pady = (gap, 0) if parent.pack_slaves() else 0
            widget.pack(fill='x', padx=padding, pady=pady)


def _appearance_color(color: Any) -> str:
    if isinstance(color, (tuple, list)):
        index = 0 if ctk.get_appearance_mode() == 'Light' else 1
        return str(color[index])
    return str(color)


def _image_color(widget: Any, color: str) -> tuple[int, int, int]:
    red, green, blue = widget.winfo_rgb(color)
    return red // 257, green // 257, blue // 257


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


def _format_path(path: tuple[str, ...]) -> str:
    return '.'.join(path)
