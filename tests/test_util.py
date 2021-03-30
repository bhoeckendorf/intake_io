import pytest

from intake_io.util import *


def test_get_axes():
    with pytest.raises(ValueError):
        get_axes(-1)
    with pytest.raises(ValueError):
        get_axes(0)
    assert get_axes(1) == "x"
    assert get_axes(2) == "yx"
    assert get_axes(3) == "zyx"
    assert get_axes(4) == "czyx"
    assert get_axes(5) == "tczyx"
    assert get_axes(6) == "itczyx"
    with pytest.raises(ValueError):
        get_axes(7)

    with pytest.raises(ValueError):
        get_axes(())
    assert get_axes((1,)) == "x"
    assert get_axes((1, 2)) == "yx"
    assert get_axes((1, 2, 9)) == "zyx"
    assert get_axes((1, 2, 3)) == "yxc"
    assert get_axes((3, 1, 2)) == "cyx"
    assert get_axes((1, 2, 3, 4)) == "czyx"
    assert get_axes((1, 2, 3, 4, 5)) == "tczyx"
    assert get_axes((1, 2, 3, 4, 5, 6)) == "itczyx"
    with pytest.raises(ValueError):
        get_axes((1, 2, 3, 4, 5, 6, 7))

    with pytest.raises(ValueError):
        get_axes(np.zeros((), np.uint8))
    assert get_axes(np.zeros((1,), np.uint8)) == "x"
    assert get_axes(np.zeros((1, 2), np.uint8)) == "yx"
    assert get_axes(np.zeros((1, 2, 9), np.uint8)) == "zyx"
    assert get_axes(np.zeros((1, 2, 3), np.uint8)) == "yxc"
    assert get_axes(np.zeros((3, 1, 2), np.uint8)) == "cyx"
    assert get_axes(np.zeros((1, 2, 3, 4), np.uint8)) == "czyx"
    assert get_axes(np.zeros((1, 2, 3, 4, 5), np.uint8)) == "tczyx"
    assert get_axes(np.zeros((1, 2, 3, 4, 5, 6), np.uint8)) == "itczyx"
    with pytest.raises(ValueError):
        get_axes(np.zeros((1, 2, 3, 4, 5, 6, 7), np.uint8))

    with pytest.raises(ValueError):
        get_axes(da.zeros((), np.uint8))
    assert get_axes(da.zeros((1,), np.uint8)) == "x"
    assert get_axes(da.zeros((1, 2), np.uint8)) == "yx"
    assert get_axes(da.zeros((1, 2, 9), np.uint8)) == "zyx"
    assert get_axes(da.zeros((1, 2, 3), np.uint8)) == "yxc"
    assert get_axes(da.zeros((3, 1, 2), np.uint8)) == "cyx"
    assert get_axes(da.zeros((1, 2, 3, 4), np.uint8)) == "czyx"
    assert get_axes(da.zeros((1, 2, 3, 4, 5), np.uint8)) == "tczyx"
    assert get_axes(da.zeros((1, 2, 3, 4, 5, 6), np.uint8)) == "itczyx"
    with pytest.raises(ValueError):
        get_axes(da.zeros((1, 2, 3, 4, 5, 6, 7), np.uint8))

    with pytest.raises(ValueError):
        get_axes(xr.DataArray((1, 2, 3)))
    with pytest.raises(ValueError):
        get_axes(xr.DataArray(np.zeros((1, 2, 3), np.uint8)))
    with pytest.raises(ValueError):
        get_axes(xr.DataArray(np.zeros((1,), np.uint8), dims=tuple("X")))
    assert get_axes(xr.DataArray(np.zeros((1,), np.uint8), dims=tuple("x"))) == "x"
    assert get_axes(xr.DataArray(np.zeros((1,), np.uint8), dims=tuple("z"))) == "z"
    assert get_axes(xr.DataArray(np.zeros((1, 2), np.uint8), dims=tuple("yx"))) == "yx"
    assert get_axes(xr.DataArray(np.zeros((1, 2, 3), np.uint8), dims=tuple("iyz"))) == "iyz"

    assert get_axes(xr.Dataset({"1": xr.DataArray(np.zeros((1,), np.uint8), dims=tuple("x"))})) == "x"
    assert get_axes(xr.Dataset({
        "1": xr.DataArray(np.zeros((8,), np.uint8), dims=tuple("x")),
        "2": xr.DataArray(np.zeros((16, 16, 8), np.uint8), dims=tuple("zyx"))
    })) == "zyx"
    assert get_axes(xr.Dataset({
        "1": xr.DataArray(np.zeros((16, 16, 8), np.uint8), dims=tuple("zyx")),
        "2": xr.DataArray(np.zeros((8,), np.uint8), dims=tuple("x"))
    })) == "zyx"


def test_get_spacing():
    assert get_spacing(xr.DataArray(np.zeros((8,)), dims=tuple("x"))) == (1,)
    assert get_spacing(xr.DataArray(np.zeros((8, 8, 8)), dims=tuple("zyx"))) == (1, 1, 1)
    assert get_spacing(xr.DataArray(np.zeros((8, 8)), dims=tuple("cx"))) == (1,)
    assert get_spacing(xr.DataArray(np.zeros((8, 8, 8, 8)), dims=tuple("czyx"))) == (1, 1, 1)
    assert get_spacing(xr.DataArray(np.zeros((8, 8, 8)), dims=tuple("zyx"), coords={
        "z": np.arange(8),
        "x": np.arange(8) * 0.125
    })) == (1, 1, 0.125)
