import numpy as np
import pytest

from pbench.preds import load_preds, save_preds


def test_roundtrip(tmp_path):
    p = tmp_path / "a" / "m.npz"
    save_preds(p, ["X", "Y"], ["g1", "g2", "g3"], np.ones((2, 3)))
    back = load_preds(p)
    assert back.perts.tolist() == ["X", "Y"]
    assert back.delta_pred.dtype == np.float32
    assert back.delta_pred.shape == (2, 3)


def test_rejects_bad_shape(tmp_path):
    with pytest.raises(ValueError, match="shape"):
        save_preds(tmp_path / "m.npz", ["X"], ["g1", "g2"], np.ones((2, 2)))


def test_rejects_nan(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        save_preds(tmp_path / "m.npz", ["X"], ["g1"], np.array([[np.nan]]))
