import numpy as np
import pytest
import torch

from scgpt_ft.model import gene_ids_for, make_model, make_vocab
from scgpt_ft.train import TrainConfig, check_split, fit, split_val, train_step

TINY = {"embsize": 16, "nheads": 2, "d_hid": 16, "nlayers": 1, "n_layers_cls": 1}


def test_split_val_is_seeded_disjoint_subset():
    perts = [f"P{i}" for i in range(50)]
    fit_, val = split_val(perts, 0.1, seed=0)
    assert len(val) == 5 and not set(fit_) & set(val) and set(fit_) | set(val) == set(perts)
    assert split_val(perts, 0.1, seed=0) == (fit_, val)


def test_check_split_rejects_overlap():
    check_split(["A"], ["B"], ["C"])
    with pytest.raises(ValueError):
        check_split(["A", "C"], ["B"], ["C"])
    with pytest.raises(ValueError):
        check_split(["A"], ["C"], ["C"])


def test_train_step_samples_max_seq_len_genes(cells):
    vocab = make_vocab(cells.genes.tolist())
    model = make_model(vocab, TINY)
    ids = gene_ids_for(cells.genes.tolist(), vocab)
    cfg = TrainConfig(max_seq_len=5, amp=False)
    batch = (torch.rand(3, 12), torch.zeros(3, 12, dtype=torch.long), torch.rand(3, 12))
    loss = train_step(model, batch, ids, cfg, "cpu")
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()


def test_fit_runs_and_returns_best_state(cells):
    vocab = make_vocab(cells.genes.tolist())
    torch.manual_seed(0)
    model = make_model(vocab, TINY)
    ids = gene_ids_for(cells.genes.tolist(), vocab)
    cfg = TrainConfig(epochs=2, max_cells=5, batch_size=4, pool_size_val=4, eval_batch_size=4,
                      amp=False, num_workers=0, patience=5)
    state, hist = fit(model, cells, ["G0", "G2"], ["G1"], ids, cfg, "cpu")
    assert len(hist) == 2 and {"epoch", "train_loss", "val_pearson", "seconds"} <= set(hist[0])
    assert set(state) == set(model.state_dict())
    assert all(np.isfinite(h["train_loss"]) for h in hist)


def test_train_step_passes_no_padding_mask(cells):
    """All genes are real tokens; an all-False mask would force PyTorch's O(L²)-memory attention."""
    vocab = make_vocab(cells.genes.tolist())
    model = make_model(vocab, TINY)
    seen = {}
    orig = model.forward

    def spy(*args, **kwargs):
        seen["mask"] = kwargs.get("src_key_padding_mask", "missing")
        return orig(*args, **kwargs)

    model.forward = spy
    batch = (torch.rand(2, 12), torch.zeros(2, 12, dtype=torch.long), torch.rand(2, 12))
    ids = gene_ids_for(cells.genes.tolist(), vocab)
    train_step(model, batch, ids, TrainConfig(amp=False), "cpu")
    assert seen["mask"] is None
