"""Tests for TIFF filename parsing and image-path resolution.

Covers both Squid filename formats:
  regular multipoint : {region}_{fov}_{z}_{channel}.tiff
  flexible multipoint: {region}_{m}_{n}_{z}_{channel}.tiff
"""
import pytest

from movie_maker_backend import FPATTERN, AcquisitionFolder


@pytest.mark.parametrize("filename,z,channel", [
    # regular multipoint (2 numeric fields: fov, z)
    ("current_0_0_Fluorescence_488_nm_Ex.tiff", 0, "Fluorescence_488_nm_Ex"),
    ("A1_0_0_BF_LED_matrix_full.tif", 0, "BF_LED_matrix_full"),
    # flexible multipoint (3 numeric fields: m, n, z)
    ("R0_0_0_0_Fluorescence_488_nm_Ex.tiff", 0, "Fluorescence_488_nm_Ex"),
    ("R0_1_2_3_Fluorescence_405_nm_Ex.tiff", 3, "Fluorescence_405_nm_Ex"),
])
def test_fpattern_extracts_channel_and_z(filename, z, channel):
    m = FPATTERN.search(filename)
    assert m is not None, f"pattern did not match {filename}"
    assert m.group("c") == channel
    assert int(m.group("z")) == z


def _make_timepoint(root, timepoint, filenames):
    tp = root / str(timepoint)
    tp.mkdir()
    for name in filenames:
        (tp / name).write_bytes(b"")  # empty stand-in; only path resolution is tested
    return tp


def test_discovers_channels_flexible_multipoint(tmp_path):
    _make_timepoint(tmp_path, 0, [
        "R0_0_0_0_Fluorescence_488_nm_Ex.tiff",
        "R0_0_0_0_BF_LED_matrix_full.tiff",
    ])
    folder = AcquisitionFolder(str(tmp_path))
    assert set(folder.channels) == {"Fluorescence_488_nm_Ex", "BF_LED_matrix_full"}


@pytest.mark.parametrize("filename", [
    "R0_0_0_0_Fluorescence_488_nm_Ex.tiff",  # flexible multipoint
    "current_0_0_Fluorescence_488_nm_Ex.tiff",  # regular multipoint
])
def test_get_image_path_resolves_both_formats(tmp_path, filename):
    _make_timepoint(tmp_path, 0, [filename])
    folder = AcquisitionFolder(str(tmp_path))
    path = folder.get_image_path(0, "Fluorescence_488_nm_Ex")
    assert path is not None
    assert path.name == filename
