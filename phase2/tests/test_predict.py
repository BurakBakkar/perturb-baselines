import numpy as np
import pytest
import torch

from scgpt_ft.model import gene_ids_for, make_model, make_vocab
from scgpt_ft.predict import forward_all_genes, predict_delta, save_preds

TINY = {"embsize": 16, "nheads": 2, "d_hid": 16, "nlayers": 1, "n_layers_cls": 1}
IDS = np.arange(12)


class EchoModel(torch.nn.Module):
    """Predicts 'no change': returns the control cell it was given."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, src, values, input_pert_flags, src_key_padding_mask, **kwargs):
        return {"mlm_output": values}


class _Batch:
    def __init__(self, x, n):
        self.x, self.pert = x, ["_"] * n

    def to(self, device):
        self.x = self.x.to(device)
        return self


def test_forward_all_genes_matches_official_pred_perturb(cells):
    vocab = make_vocab(cells.genes.tolist())
    torch.manual_seed(0)
    model = make_model(vocab, TINY)
    ids = gene_ids_for(cells.genes.tolist(), vocab)
    vals = torch.rand(3, 12)
    flags = torch.zeros(3, 12)
    flags[:, 2] = 1
    x = torch.stack([vals.reshape(-1), flags.reshape(-1)], dim=1)
    official = model.pred_perturb(_Batch(x, 3), include_zero_gene="all", gene_ids=ids, amp=False)
    ours = forward_all_genes(model, vals, flags.long(), ids, amp=False)
    assert torch.allclose(ours, official.float(), atol=1e-6)


def test_forward_all_genes_passes_no_padding_mask():
    seen = {}

    class Spy(EchoModel):
        def forward(self, src, values, input_pert_flags, src_key_padding_mask, **kwargs):
            seen["mask"] = src_key_padding_mask
            return super().forward(src, values, input_pert_flags, src_key_padding_mask)

    forward_all_genes(Spy(), torch.rand(2, 12), torch.zeros(2, 12, dtype=torch.long), IDS, False)
    assert seen["mask"] is None


def test_predict_delta_is_zero_for_identity_model(cells):
    n_ctrl = len(cells.ctrl_rows)
    d = predict_delta(EchoModel(), cells, ["G0", "G1"], IDS, pool_size=n_ctrl, batch_size=7,
                      seed=0, amp=False)
    assert d.shape == (2, 12) and d.dtype == np.float32
    assert np.allclose(d, 0.0, atol=1e-5)


def test_predict_pool_larger_than_controls(cells):
    d = predict_delta(EchoModel(), cells, ["G0"], IDS, pool_size=500, batch_size=64,
                      seed=0, amp=False)
    assert np.isfinite(d).all()


def test_predict_is_seeded_and_order_independent(cells):
    a = predict_delta(EchoModel(), cells, ["G0", "G2"], IDS, 10, 4, seed=0, amp=False)
    b = predict_delta(EchoModel(), cells, ["G2", "G0"], IDS, 10, 4, seed=0, amp=False)
    assert np.array_equal(a[0], b[1]) and np.array_equal(a[1], b[0])


def test_predict_unmeasured_target_is_finite(cells):
    vocab = make_vocab(cells.genes.tolist())
    model = make_model(vocab, TINY)
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


def test_predict_releases_cached_gpu_memory_before_and_after(cells, monkeypatch):
    """Training and inference caches together exceed 16 GB; on WSL that spills to host RAM."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    predict_delta(EchoModel(), cells, ["G0"], IDS, 4, 4, seed=0, amp=False)
    assert len(calls) == 2


def test_subset_rows_follows_requested_order():
    from scgpt_ft.predict import subset_rows

    delta = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = subset_rows(["A", "B", "C"], delta, ["C", "A"])
    assert np.array_equal(out, delta[[2, 0]])
    with pytest.raises(KeyError):
        subset_rows(["A"], delta[:1], ["Z"])
