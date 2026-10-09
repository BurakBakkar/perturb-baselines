"""scGPT's TransformerGenerator, initialised from the whole-human checkpoint as in the tutorial."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab

SPECIAL_TOKENS = ["<pad>", "<cls>", "<eoc>"]
# Tutorial_Perturbation loads only these; the perturbation encoder and decoder start fresh.
LOAD_PREFIXES = ("encoder", "value_encoder", "transformer_encoder")


def _with_specials(vocab: GeneVocab) -> GeneVocab:
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    return vocab


def load_vocab(vocab_file) -> GeneVocab:
    return _with_specials(GeneVocab.from_file(vocab_file))


def make_vocab(genes) -> GeneVocab:
    return _with_specials(GeneVocab(list(genes) + SPECIAL_TOKENS))


def gene_ids_for(genes, vocab: GeneVocab) -> np.ndarray:
    pad = vocab["<pad>"]
    return np.array([vocab[g] if g in vocab else pad for g in genes], dtype=int)


def make_model(vocab: GeneVocab, args: dict) -> TransformerGenerator:
    return TransformerGenerator(
        len(vocab), args["embsize"], args["nheads"], args["d_hid"], args["nlayers"],
        nlayers_cls=args["n_layers_cls"], n_cls=1, vocab=vocab, dropout=0.0,
        pad_token="<pad>", pad_value=0, pert_pad_id=0,
        use_fast_transformer=False,  # flash-attn v1 is not built; plain PyTorch attention
    )


def remap_flash_keys(state_dict: dict) -> dict:
    """flash-attn's fused Wqkv projection == nn.MultiheadAttention's in_proj."""
    return {k.replace("self_attn.Wqkv.", "self_attn.in_proj_"): v for k, v in state_dict.items()}


def load_pretrained(model: TransformerGenerator, state_dict: dict) -> list[str]:
    """Copy every LOAD_PREFIXES weight; raise if any of the model's such weights is not covered."""
    sd = {k: v for k, v in remap_flash_keys(state_dict).items() if k.startswith(LOAD_PREFIXES)}
    own = model.state_dict()
    missing = [k for k in own if k.startswith(LOAD_PREFIXES) and k not in sd]
    if missing:
        raise RuntimeError(f"model weights not in the checkpoint: {missing[:5]}")
    unexpected = [k for k in sd if k not in own]
    bad_shape = [k for k in sd if k in own and own[k].shape != sd[k].shape]
    if unexpected or bad_shape:
        raise RuntimeError(f"checkpoint keys unknown to the model: {unexpected[:5]}; "
                           f"shape mismatch: {bad_shape[:5]}")
    own.update(sd)
    model.load_state_dict(own)
    return sorted(sd)


def build_model(ckpt_dir, genes, device) -> tuple[TransformerGenerator, np.ndarray]:
    ckpt_dir = Path(ckpt_dir)
    vocab = load_vocab(ckpt_dir / "vocab.json")
    args = json.loads((ckpt_dir / "args.json").read_text())
    model = make_model(vocab, args)
    state = torch.load(ckpt_dir / "best_model.pt", map_location="cpu")
    loaded = load_pretrained(model, state)
    print(f"[model] loaded {len(loaded)} pretrained tensors", flush=True)
    return model.to(device), gene_ids_for(list(genes), vocab)
