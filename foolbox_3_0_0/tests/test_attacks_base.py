from typing import Tuple
import pytest
import eagerpy as ep
from foolbox_3_0_0 import foolbox as fbn

attacks = [
    foolbox_3_0_0.foolbox.attacks.InversionAttack(distance=foolbox_3_0_0.foolbox.distances.l2),
    foolbox_3_0_0.foolbox.attacks.InversionAttack(distance=foolbox_3_0_0.foolbox.distances.l2).repeat(3),
    foolbox_3_0_0.foolbox.attacks.L2ContrastReductionAttack(),
    foolbox_3_0_0.foolbox.attacks.L2ContrastReductionAttack().repeat(3),
]


@pytest.mark.parametrize("attack", attacks)
def test_call_one_epsilon(
    fmodel_and_data_ext_for_attacks: Tuple[
        Tuple[fbn.Model, ep.Tensor, ep.Tensor], bool
    ],
    attack: fbn.Attack,
) -> None:
    (fmodel, x, y), _ = fmodel_and_data_ext_for_attacks

    assert ep.istensor(x)
    assert ep.istensor(y)

    raw, clipped, success = attack(fmodel, x, y, epsilons=1.0)
    assert ep.istensor(raw)
    assert ep.istensor(clipped)
    assert ep.istensor(success)
    assert raw.shape == x.shape
    assert clipped.shape == x.shape
    assert success.shape == (len(x),)


def test_get_channel_axis() -> None:
    class Model:
        data_format = None

    model = Model()
    model.data_format = "channels_first"  # type: ignore
    assert foolbox_3_0_0.foolbox.attacks.base.get_channel_axis(model, 3) == 1  # type: ignore
    model.data_format = "channels_last"  # type: ignore
    assert foolbox_3_0_0.foolbox.attacks.base.get_channel_axis(model, 3) == 2  # type: ignore
    model.data_format = "invalid"  # type: ignore
    with pytest.raises(ValueError):
        assert foolbox_3_0_0.foolbox.attacks.base.get_channel_axis(model, 3)  # type: ignore
