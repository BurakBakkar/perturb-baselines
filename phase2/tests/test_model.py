import numpy as np
import pytest
import torch

from scgpt_ft.model import (
    gene_ids_for,
    load_pretrained,
    make_model,
    make_vocab,
    remap_flash_keys,
)

TINY = {"embsize": 16, "nheads": 2, "d_hid": 16, "nlayers": 1, "n_layers_cls": 1}


def _flash_style(sd):
    """What the real checkpoint looks like: flash-attn Wqkv names, plus heads we don't load."""
    out = {k.replace("self_attn.in_proj_", "self_attn.Wqkv."): v.clone() for k, v in sd.items()}
    out["flag_encoder.weight"] = torch.zeros(3, 16)
    out["mvc_decoder.W.weight"] = torch.zeros(16, 16)
    return out


def test_remap_flash_keys():
    sd = {"transformer_encoder.layers.0.self_attn.Wqkv.weight": 1,
          "transformer_encoder.layers.0.self_attn.Wqkv.bias": 2, "encoder.x": 3}
    assert remap_flash_keys(sd) == {
        "transformer_encoder.layers.0.self_attn.in_proj_weight": 1,
        "transformer_encoder.layers.0.self_attn.in_proj_bias": 2, "encoder.x": 3}


def test_gene_ids_map_unknown_to_pad():
    vocab = make_vocab(["A", "B"])
    ids = gene_ids_for(["A", "ZZZ", "B"], vocab)
    assert ids[1] == vocab["<pad>"] and ids[0] != ids[2]


def test_load_pretrained_copies_prefix_weights_only():
    vocab = make_vocab([f"g{i}" for i in range(12)])
    torch.manual_seed(0)
    src = make_model(vocab, TINY)
    torch.manual_seed(1)
    dst = make_model(vocab, TINY)
    loaded = load_pretrained(dst, _flash_style(src.state_dict()))
    assert any("in_proj_weight" in k for k in loaded)
    for k, v in dst.state_dict().items():
        if k.startswith(("encoder.", "value_encoder.", "transformer_encoder.")):
            assert torch.equal(v, src.state_dict()[k]), k
    # perturbation encoder and decoder are new: not copied
    assert not torch.equal(dst.pert_encoder.weight, src.pert_encoder.weight)


def test_load_pretrained_raises_on_unloaded_prefix_weight():
    vocab = make_vocab([f"g{i}" for i in range(12)])
    src = make_model(vocab, TINY)
    sd = _flash_style(src.state_dict())
    del sd["transformer_encoder.layers.0.self_attn.Wqkv.weight"]
    with pytest.raises(RuntimeError, match="not in the checkpoint"):
        load_pretrained(make_model(vocab, TINY), sd)


def test_load_pretrained_raises_on_shape_mismatch():
    vocab = make_vocab([f"g{i}" for i in range(12)])
    sd = _flash_style(make_model(vocab, TINY).state_dict())
    sd["encoder.embedding.weight"] = torch.zeros(3, 16)
    with pytest.raises(RuntimeError, match="shape"):
        load_pretrained(make_model(vocab, TINY), sd)


def test_tiny_model_forward():
    vocab = make_vocab([f"g{i}" for i in range(12)])
    model = make_model(vocab, TINY)
    ids = torch.as_tensor(gene_ids_for([f"g{i}" for i in range(12)], vocab)).repeat(2, 1)
    vals = torch.rand(2, 12)
    flags = torch.zeros(2, 12, dtype=torch.long)
    out = model(ids, vals, flags, src_key_padding_mask=torch.zeros(2, 12, dtype=torch.bool),
                CLS=False, CCE=False, MVC=False, ECS=False)
    assert out["mlm_output"].shape == (2, 12)
    assert np.isfinite(out["mlm_output"].detach().numpy()).all()
