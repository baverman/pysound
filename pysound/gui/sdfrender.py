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
    # Keep source coordinates one-dimensional until a full field is required.
    PX = X[0].copy()
    PY = Y[:, 0].copy()
    PX -= center_x
    PY -= center_y

    hs = sweep / 2
    hl = math.cos(hs)
    m_ = (math.cos(start + hs), math.sin(start + hs))

    # Distance from the arc center.
    d = (PY * PY)[:, None] + (PX * PX)[None, :]
    np.sqrt(d, out=d)

    # Angular sector test based on the arc midpoint direction.
    n = (PY * m_[1])[:, None] + (PX * m_[0])[None, :]
    inside = n >= d * hl

    np.subtract(d, radius, out=d)
    np.abs(d, out=d)
    circle_dist = d

    start_x = radius * math.cos(start)
    start_y = radius * math.sin(start)
    end_x = radius * math.cos(start + sweep)
    end_y = radius * math.sin(start + sweep)

    # Squared distance to the start endpoint. Reuse n as the output field.
    work_x = PX - start_x
    work_y = PY - start_y
    work_x *= work_x
    work_y *= work_y
    np.add(work_y[:, None], work_x[None, :], out=n)

    # Squared distance to the end endpoint.
    work_x = PX - end_x
    work_y = PY - end_y
    work_x *= work_x
    work_y *= work_y
    endpoint_dist = work_y[:, None] + work_x[None, :]
    np.minimum(endpoint_dist, n, out=endpoint_dist)
    np.sqrt(endpoint_dist, out=endpoint_dist)

    np.copyto(endpoint_dist, circle_dist, where=inside)
    endpoint_dist -= half_width
    return endpoint_dist

__all__ = ['arc', 'circle', 'composite', 'coverage', 'segment']
