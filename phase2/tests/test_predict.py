import numpy as np
import pytest

from scgpt_ft.model import gene_ids_for, make_model, make_vocab
from scgpt_ft.predict import predict_delta, save_preds


class EchoModel:
    """Predicts 'no change': returns the control cell it was given."""

    def pred_perturb(self, batch, include_zero_gene, gene_ids, amp):
        n = len(batch.pert)
        return batch.x[:, 0].view(n, -1)


def test_predict_delta_is_zero_for_identity_model(cells):
    n_ctrl = len(cells.ctrl_rows)
    d = predict_delta(EchoModel(), cells, ["G0", "G1"], None, pool_size=n_ctrl, batch_size=7,
                      seed=0, amp=False)
    assert d.shape == (2, 12) and d.dtype == np.float32
    assert np.allclose(d, 0.0, atol=1e-5)


def test_predict_pool_larger_than_controls(cells):
    d = predict_delta(EchoModel(), cells, ["G0"], None, pool_size=500, batch_size=64,
                      seed=0, amp=False)
    assert np.isfinite(d).all()


def test_predict_is_seeded_and_order_independent(cells):
    a = predict_delta(EchoModel(), cells, ["G0", "G2"], None, 10, 4, seed=0, amp=False)
    b = predict_delta(EchoModel(), cells, ["G2", "G0"], None, 10, 4, seed=0, amp=False)
    assert np.array_equal(a[0], b[1]) and np.array_equal(a[1], b[0])


def test_predict_unmeasured_target_is_finite(cells):
    vocab = make_vocab(cells.genes.tolist())
    model = make_model(vocab, {"embsize": 16, "nheads": 2, "d_hid": 16, "nlayers": 1,
                               "n_layers_cls": 1})
    ids = gene_ids_for(cells.genes.tolist(), vocab)
    d = predict_delta(model, cells, ["G0", "NOTMEASURED"], ids, 8, 4, seed=0, amp=False)
    assert np.isfinite(d).all()


def test_save_preds_contract(tmp_path):
    save_preds(tmp_path / "m.npz", ["A", "B"], ["G0", "G1", "G2"], np.zeros((2, 3)))
    with np.load(tmp_path / "m.npz", allow_pickle=False) as z:
        assert z["perts"].tolist() == ["A", "B"]
        assert z["delta_pred"].dtype == np.float32
    with pytest.raises(ValueError):
        save_preds(tmp_path / "bad.npz", ["A"], ["G0"], np.full((1, 1), np.nan))
    with pytest.raises(ValueError):
        save_preds(tmp_path / "bad.npz", ["A"], ["G0", "G1"], np.zeros((1, 1)))
