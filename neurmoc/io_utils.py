"""File IO helpers: MATLAB .mat (v7 and v7.3) loading, path checks.

`load_mat` reads v7.3 HDF5 and earlier MATLAB formats. Arrays are returned
as NumPy arrays in MATLAB axis order.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def require_dir(path, label: str = "Required"):
    """Return `path` if it is an existing directory, else raise."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} directory not found: {path}")
    return Path(path)


def require_file(path, label: str = "Required"):
    """Return `path` if it is an existing file, else raise."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} file not found: {path}")
    return Path(path)


def ensure_dir(path):
    """Create `path` (and parents) if needed and return it."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def _is_hdf5(path) -> bool:
    # MATLAB v7.3 files are HDF5 with a 512-byte user block, so the HDF5
    # signature sits at offset 512 (plain HDF5 files have it at offset 0).
    signature = b"\x89HDF\r\n\x1a\n"
    with open(path, "rb") as fh:
        if fh.read(8) == signature:
            return True
        fh.seek(512)
        return fh.read(8) == signature


def load_mat(path, variables: Iterable[str] | None = None) -> dict:
    """Load a MATLAB .mat file of any version into a dict of numpy arrays.

    v7.3 files (HDF5) are read with h5py and every dataset is transposed so
    the axis order matches what MATLAB (and scipy.io.loadmat) would report.
    Pass `variables` to read only a subset (saves time/memory on big files).
    """
    path = require_file(path, "MAT")
    if _is_hdf5(path):
        import h5py

        out = {}
        with h5py.File(path, "r") as fh:
            names = variables if variables is not None else list(fh.keys())
            for name in names:
                if name.startswith("#"):
                    continue
                node = fh[name]
                if isinstance(node, h5py.Dataset):
                    arr = node[()]
                    if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                        arr = arr.T  # undo MATLAB column-major storage
                    out[name] = arr
        return out

    import scipy.io as sio

    raw = sio.loadmat(path, variable_names=list(variables) if variables else None)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def save_mat(path, data: Mapping[str, np.ndarray]) -> None:
    """Save a dict of arrays to a .mat file (scipy v5 format)."""
    import scipy.io as sio

    sio.savemat(path, dict(data))


def load_npz_or_mat(path_stem, variables: Iterable[str] | None = None) -> dict:
    """Load `<stem>.npz` if it exists, else `<stem>.mat` (any version).

    The .npz file takes precedence when both formats exist.
    """
    stem = Path(path_stem)
    npz_path = stem.with_suffix(".npz")
    if npz_path.is_file():
        with np.load(npz_path) as data:
            names = variables if variables is not None else data.files
            return {name: data[name] for name in names}
    mat_path = stem.with_suffix(".mat")
    if mat_path.is_file():
        return load_mat(mat_path, variables)
    raise FileNotFoundError(f"Neither {npz_path} nor {mat_path} exists")
