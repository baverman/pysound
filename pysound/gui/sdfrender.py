from __future__ import annotations

import math

import numpy as np


def coverage(distance: np.ndarray) -> np.ndarray:
    # smooth(?) version
    # alpha = np.clip(0.5 - distance, 0.0, 1.0)
    # return alpha * alpha * (3 - 2 * alpha)
    return np.clip(0.5 - distance, 0.0, 1.0)


def composite(
    image: np.ndarray,
    color: tuple[int, int, int],
    alpha: np.ndarray,
) -> None:
    opacity = alpha[..., None]
    image *= 1 - opacity
    image += np.asarray(color, dtype=np.float32) * opacity


def _ref_arc(
    X: np.ndarray,
    Y: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    half_width: float,
    start: float,
    sweep: float,
) -> np.ndarray:
    # notation:
    # lowercase: scalar
    # lowercase_: vector (2,)
    # uppercase: matrix (N, M)
    a = sweep / 2

    # local unit vectors
    ly_ = (math.cos(start + a), math.sin(start + a))
    lx_ = (-ly_[1], ly_[0])

    # arc end and its normal unit vectors towards middle point in local coordinates
    le_ = (math.sin(a), math.cos(a))
    ln_ = (-le_[1], le_[0])

    PX = X - center_x
    PY = Y - center_y
    lPX = np.abs(PX * lx_[0] + PY * lx_[1])  # reflect over local y
    lPY = PX * ly_[0] + PY * ly_[1]

    sector = lPX * ln_[0] + lPY * ln_[1]
    circle_dist = np.abs(np.hypot(PX, PY) - radius)
    endpoint_dist = np.hypot(lPX - le_[0] * radius, lPY - le_[1] * radius)
    distance = endpoint_dist
    np.copyto(distance, circle_dist, where=sector >= 0)
    return distance - half_width


def segment(
    x: np.ndarray,
    y: np.ndarray,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    half_width: float,
) -> np.ndarray:
   dx = np.float32(end_x - start_x)
   dy = np.float32(end_y - start_y)

   px = x[0].copy()
   px -= start_x

   py = y[:, 0].copy()
   py -= start_y

   # Projection position along the segment.
   length_squared = dx*dx + dy*dy
   project_x = dx / length_squared
   project_y = dy / length_squared
   position = (py * project_y)[:, None] + (px * project_x)[None, :]
   np.clip(position, 0.0, 1.0, out=position)

   # Squared horizontal distance.
   result = position.copy()
   result *= dx
   result -= px[None, :]
   result *= result

   # Add squared vertical distance. Reuse position as work space.
   position *= dy
   position -= py[:, None]
   position *= position
   result += position

   np.sqrt(result, out=result)
   result -= half_width
   return result


def circle(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
) -> np.ndarray:
    dX = x[0].copy()
    dX -= center_x
    dX *= dX

    dY = y[:,0].copy()
    dY -= center_y
    dY *= dY

    res = dX[None,:] + dY[:,None]
    np.sqrt(res, out=res)
    res -= radius
    return res

def arc(
    X: np.ndarray,
    Y: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
    half_width: float,
    start: float,
    sweep: float,
) -> np.ndarray:
    # Keep grid coordinates separated until a full 2D field is required.
    x = X[0]
    y = Y[:, 0]
    PX = x - center_x
    PY = y - center_y

    start_ = (math.cos(start), math.sin(start))
    end_angle = start + sweep
    end_ = (math.cos(end_angle), math.sin(end_angle))

    # Distance to the full circle.
    work_x = PX*PX
    work_y = PY*PY
    circle_dist = work_y[:, None] + work_x[None, :]
    np.sqrt(circle_dist, out=circle_dist)
    circle_dist -= radius
    np.abs(circle_dist, out=circle_dist)

    # Sector half-planes. Positive sweep is counterclockwise.
    direction = 1.0 if sweep >= 0.0 else -1.0
    np.multiply(PX, -direction * start_[1], out=work_x)
    np.multiply(PY, direction * start_[0], out=work_y)
    half_plane = work_y[:, None] + work_x[None, :]
    inside = half_plane >= 0.0

    np.multiply(PX, direction * end_[1], out=work_x)
    np.multiply(PY, -direction * end_[0], out=work_y)
    np.add(work_y[:, None], work_x[None, :], out=half_plane)
    if abs(sweep) <= math.pi:
        np.logical_and(inside, half_plane >= 0.0, out=inside)
    else:
        np.logical_or(inside, half_plane >= 0.0, out=inside)

    # Squared distance to the start endpoint.
    endpoint_x = center_x + radius * start_[0]
    endpoint_y = center_y + radius * start_[1]
    np.subtract(x, endpoint_x, out=work_x)
    np.subtract(y, endpoint_y, out=work_y)
    work_x *= work_x
    work_y *= work_y
    endpoint_dist = work_y[:, None] + work_x[None, :]

    # Squared distance to the end endpoint. Reuse half_plane as work space.
    endpoint_x = center_x + radius * end_[0]
    endpoint_y = center_y + radius * end_[1]
    np.subtract(x, endpoint_x, out=work_x)
    np.subtract(y, endpoint_y, out=work_y)
    work_x *= work_x
    work_y *= work_y
    np.add(work_y[:, None], work_x[None, :], out=half_plane)
    np.minimum(endpoint_dist, half_plane, out=endpoint_dist)
    np.sqrt(endpoint_dist, out=endpoint_dist)

    # Use circle distance inside the sector and endpoint distance outside it.
    np.copyto(endpoint_dist, circle_dist, where=inside)
    endpoint_dist -= half_width
    return endpoint_dist

__all__ = ['arc', 'circle', 'composite', 'coverage', 'segment']
