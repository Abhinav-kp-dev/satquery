import os
import sys
import tempfile
from pathlib import Path

# Isolated storage for the whole test session; must be set before `app` is imported.
_tmp = tempfile.mkdtemp(prefix="satquery-test-")
os.environ["SATQUERY_STORAGE_DIR"] = _tmp
os.environ["SATQUERY_VLM_PROVIDER"] = "mock"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from app.config import SAMPLE_DIR  # noqa: E402
from app.samples.generate import generate  # noqa: E402


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    return generate(SAMPLE_DIR)


def write_tif(path: Path, data: np.ndarray, crs="EPSG:32646", origin=(368_000.0, 2_900_000.0), gsd=10.0, names=None, date=None):
    with rasterio.open(
        path, "w", driver="GTiff", width=data.shape[2], height=data.shape[1], count=data.shape[0],
        dtype=str(data.dtype), crs=crs, transform=from_origin(origin[0], origin[1], gsd, gsd),
    ) as dst:
        dst.write(data)
        for i, n in enumerate(names or [], start=1):
            dst.set_band_description(i, n)
        if date:
            dst.update_tags(ACQUISITION_DATE=date)
    return path
