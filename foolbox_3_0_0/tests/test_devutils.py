import pytest
import eagerpy as ep


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_atleast_kd_1d(dummy: ep.Tensor, k: int) -> None:
    x = ep.zeros(dummy, (10,))
    x = foolbox_3_0_0.foolbox.devutils.atleast_kd(x, k)
    assert x.shape[0] == 10
    assert x.ndim == k


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_atleast_kd_3d(dummy: ep.Tensor, k: int) -> None:
    x = ep.zeros(dummy, (10, 5, 3))
    x = foolbox_3_0_0.foolbox.devutils.atleast_kd(x, k)
    assert x.shape[:3] == (10, 5, 3)
    assert x.ndim == max(k, 3)


def test_flatten_2d(dummy: ep.Tensor) -> None:
    x = ep.zeros(dummy, (4, 5))
    x = foolbox_3_0_0.foolbox.devutils.flatten(x)
    assert x.shape == (4, 5)


def test_flatten_3d(dummy: ep.Tensor) -> None:
    x = ep.zeros(dummy, (4, 5, 6))
    x = foolbox_3_0_0.foolbox.devutils.flatten(x)
    assert x.shape == (4, 30)


def test_flatten_4d(dummy: ep.Tensor) -> None:
    x = ep.zeros(dummy, (4, 5, 6, 7))
    x = foolbox_3_0_0.foolbox.devutils.flatten(x)
    assert x.shape == (4, 210)
