import pytest
import eagerpy as ep


def test_plot(dummy: ep.Tensor) -> None:
    # just tests that the calls don't throw any errors
    images = ep.zeros(dummy, (10, 3, 32, 32))
    foolbox_3_0_0.foolbox.plot.images(images)
    foolbox_3_0_0.foolbox.plot.images(images, n=3)
    foolbox_3_0_0.foolbox.plot.images(images, n=3, data_format="channels_first")
    foolbox_3_0_0.foolbox.plot.images(images, nrows=4)
    foolbox_3_0_0.foolbox.plot.images(images, ncols=3)
    foolbox_3_0_0.foolbox.plot.images(images, nrows=2, ncols=6)
    foolbox_3_0_0.foolbox.plot.images(images, nrows=2, ncols=4)
    with pytest.raises(ValueError):
        images = ep.zeros(dummy, (10, 3, 3, 3))
        foolbox_3_0_0.foolbox.plot.images(images)
    with pytest.raises(ValueError):
        images = ep.zeros(dummy, (10, 1, 1, 1))
        foolbox_3_0_0.foolbox.plot.images(images)
    with pytest.raises(ValueError):
        images = ep.zeros(dummy, (10, 32, 32))
        foolbox_3_0_0.foolbox.plot.images(images)
    with pytest.raises(ValueError):
        images = ep.zeros(dummy, (10, 3, 32, 32))
        foolbox_3_0_0.foolbox.plot.images(images, data_format="foo")
