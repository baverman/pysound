from __future__ import annotations

import math

import numpy as np


def coverage(distance: np.ndarray) -> np.ndarray:
    alpha = np.clip(0.5 - distance, 0.0, 1.0)
    return alpha * alpha * (3 - 2 * alpha)


def composite(
    image: np.ndarray,
    color: tuple[int, int, int],
    alpha: np.ndarray,
) -> None:
    opacity = alpha[..., None]
    image *= 1 - opacity
    image += np.asarray(color, dtype=np.float32) * opacity


def arc(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    half_width: float,
    start: float,
    sweep: float,
) -> np.ndarray:
    dx = x - center_x
    dy = y - center_y
    middle = start + sweep * 0.5
    middle_cos = math.cos(middle)
    middle_sin = math.sin(middle)
    local_x = np.abs(-middle_sin * dx + middle_cos * dy)
    local_y = middle_cos * dx + middle_sin * dy

    aperture = sweep * 0.5
    aperture_sin = math.sin(aperture)
    aperture_cos = math.cos(aperture)
    endpoint_distance = np.hypot(
        local_x - aperture_sin * radius,
        local_y - aperture_cos * radius,
    )
    radial_distance = np.abs(np.hypot(dx, dy) - radius)
    outside_aperture = aperture_cos * local_x > aperture_sin * local_y
    centerline_distance = np.where(
        outside_aperture,
        endpoint_distance,
        radial_distance,
    )
    return centerline_distance - half_width


def segment(
    x: np.ndarray,
    y: np.ndarray,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    half_width: float,
) -> np.ndarray:
    dx = end_x - start_x
    dy = end_y - start_y
    position = np.clip(
        ((x - start_x) * dx + (y - start_y) * dy) / (dx * dx + dy * dy),
        0.0,
        1.0,
    )
    nearest_x = start_x + position * dx
    nearest_y = start_y + position * dy
    return np.hypot(x - nearest_x, y - nearest_y) - half_width


def circle(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
) -> np.ndarray:
    return np.hypot(x - center_x, y - center_y) - radius


__all__ = ['arc', 'circle', 'composite', 'coverage', 'segment']
