"""Cheap, deterministic spatial description of a mask's centroid, used for
grounding answers ("the northeast quadrant") without any detection model.
"""
from __future__ import annotations

import numpy as np

_ROWS = ("north", "central", "south")
_COLS = ("west", "central", "east")


def region_hint(mask: np.ndarray) -> str:
    """Returns a phrase that reads naturally after 'concentrated in ...',
    e.g. 'the southeast of the scene', 'the centre of the scene', or the
    bare fallback 'the scene' when no mask is available."""
    if mask is None or not mask.any():
        return "the scene"
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    row_frac = float(ys.mean()) / max(h - 1, 1)
    col_frac = float(xs.mean()) / max(w - 1, 1)
    row = _ROWS[min(int(row_frac * 3), 2)]
    col = _COLS[min(int(col_frac * 3), 2)]
    if row == "central" and col == "central":
        return "the centre of the scene"
    if row == "central":
        return f"the {col} of the scene"
    if col == "central":
        return f"the {row} of the scene"
    return f"the {row}{col} of the scene"
