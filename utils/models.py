"""Load models and extract their embedding matrices."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval().to(device)

    # GPT-2 has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_final_norm(model):
    """
    Return the final layer-norm module applied just before the LM head.
    This must be applied to raw hidden states before cosine-similarity
    comparisons — the raw residual stream has a very different direction
    to what the LM head actually sees.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f          # GPT-2
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return model.model.decoder.final_layer_norm  # OPT
    raise ValueError(f"Unknown architecture: {type(model)}")


def get_embedding_matrix(model, device: str) -> torch.Tensor:
    """
    Return the input embedding matrix E of shape (vocab_size, d).
    Works for GPT-2 (wte) and OPT (embed_tokens).
    """
    # GPT-2 family
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        emb = model.transformer.wte.weight.detach()
    # OPT family
    elif hasattr(model, "model") and hasattr(model.model, "decoder"):
        emb = model.model.decoder.embed_tokens.weight.detach()
    else:
        raise ValueError(f"Unknown architecture: {type(model)}")

    return emb.to(device)
