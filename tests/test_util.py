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
    assert get_spacing(xr.DataArray(np.zeros((8,)), dims=tuple("x"))) == (None,)
    assert get_spacing(xr.DataArray(np.zeros((8,)), dims=tuple("x")), "x") == None
    assert get_spacing(xr.DataArray(np.zeros((8, 8, 8)), dims=tuple("zyx"))) == (None, None, None)
    assert get_spacing(xr.DataArray(np.zeros((8, 8)), dims=tuple("cx"))) == (None,)
    assert get_spacing(xr.DataArray(np.zeros((8, 8, 8, 8)), dims=tuple("czyx"))) == (None, None, None)
    assert get_spacing(xr.DataArray(np.zeros((8, 8, 8)), dims=tuple("zyx"), coords={
        "z": np.arange(8),
        "x": np.arange(8) * 0.125
    })) == (1, None, 0.125)

    assert get_spacing(xr.Dataset({
        "image": xr.DataArray(np.zeros((8, 8, 8)), dims=tuple("zyx"), coords={
            "z": np.arange(8),
            "x": np.arange(8) * 0.125
        })
    })) == (1, None, 0.125)
    assert get_spacing(xr.Dataset({
        "0": xr.DataArray(np.zeros((8, 8)), dims=tuple("yx")),
        "image": xr.DataArray(np.zeros((8, 8, 8)), dims=tuple("zyx"), coords={
            "z": np.arange(8),
            "x": np.arange(8) * 0.125
        })
    })) == (1, None, 0.125)


def test_to_xarray():
    arr = xr.DataArray(np.zeros((8, 8, 8)), dims=tuple("zyx"), coords={
        "z": np.arange(8),
        "x": np.arange(8) * 0.125
    })
    with pytest.raises(ValueError):
        to_xarray(xr.Dataset({"image": arr, "0": arr}))
    assert isinstance(to_xarray(xr.Dataset({"image": arr})), xr.DataArray)
    assert to_xarray(xr.Dataset({"image": arr})).shape == arr.shape
    assert get_spacing(to_xarray(xr.Dataset({"image": arr}))) == get_spacing(arr)

    arr = to_xarray(np.zeros((8, 2, 16, 32), np.uint8), (0.1, 0.2, 0.3), axes="tcyx", coords={"c": (1, 2)})
    assert get_spacing(arr) == (0.1, 0.2, 0.3)
    assert arr.coords["t"][1] == 0.1
    assert arr.coords["c"][1] == 2
    assert arr.coords["y"][1] == 0.2
    assert arr.coords["x"][1] == 0.3

    arr = to_xarray(np.zeros((8, 2, 16, 32), np.uint8), (0.2, 0.3), axes="tcyx", coords={"c": (1, 2)})
    assert get_spacing(arr) == (None, 0.2, 0.3)
    assert "t" not in arr.coords
    assert arr.coords["c"][1] == 2
    assert arr.coords["y"][1] == 0.2
    assert arr.coords["x"][1] == 0.3

    arr = to_xarray(np.zeros((8, 2, 16, 32), np.uint8), (0.2, 0.3), axes="tcyx")
    assert get_spacing(arr) == (None, 0.2, 0.3)
    assert "t" not in arr.coords
    assert "c" not in arr.coords
    assert arr.coords["y"][1] == 0.2
    assert arr.coords["x"][1] == 0.3

    arr = to_xarray(np.zeros((2, 8, 2, 8, 16, 32), np.uint8), (0.1, 0.2, 0.3), axes="itczyx", coords={"c": (1, 2)})
    assert get_spacing(arr) == (None, 0.1, 0.2, 0.3)
    for i in "it":
        assert i not in arr.coords
    assert arr.coords["c"][1] == 2
    assert arr.coords["z"][1] == 0.1
    assert arr.coords["y"][1] == 0.2
    assert arr.coords["x"][1] == 0.3

    arr = to_xarray(np.zeros((2, 8, 2, 8, 16, 32), np.uint8), (0.1, 0.4, 0.2, 0.3), axes="itczyx", coords={"c": (1, 2)})
    assert get_spacing(arr) == (0.1, 0.4, 0.2, 0.3)
    assert "i" not in arr.coords
    assert arr.coords["t"][1] == 0.1
    assert arr.coords["c"][1] == 2
    assert arr.coords["z"][1] == 0.4
    assert arr.coords["y"][1] == 0.2
    assert arr.coords["x"][1] == 0.3
