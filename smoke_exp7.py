"""Throwaway smoke test for exp7 shared-embedding builder. Delete after use."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

log = open("exp7_smoke_log.txt", "w", encoding="utf-8")
def L(*a):
    print(*a, file=log, flush=True)

L("start")
import torch
import torch.nn.functional as F
L("torch", torch.__version__)
from transformers import AutoTokenizer, AutoModel
L("transformers imported")

gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
L("gpt2 tokenizer")
bert_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
L("bert tokenizer")
bert     = AutoModel.from_pretrained("distilbert-base-uncased")
bert_emb = bert.embeddings.word_embeddings.weight.detach().clone()
del bert
L("bert embeddings", tuple(bert_emb.shape))

vocab_size = len(gpt2_tok)
BATCH = 512
E = torch.empty(vocab_size, 768, dtype=torch.float32)
t0 = time.time()
for start in range(0, vocab_size, BATCH):
    ids = list(range(start, min(start + BATCH, vocab_size)))
    strings = gpt2_tok.batch_decode([[i] for i in ids], clean_up_tokenization_spaces=False)
    enc = bert_tok(strings, add_special_tokens=False, padding=True, return_tensors="pt")
    sub_ids, sub_mask = enc["input_ids"], enc["attention_mask"]
    if sub_ids.shape[1] == 0:   # all-empty batch: pad to 1 col of UNK
        sub_ids  = torch.full((len(ids), 1), bert_tok.unk_token_id, dtype=torch.long)
        sub_mask = torch.zeros(len(ids), 1, dtype=torch.long)
    sub_emb = bert_emb[sub_ids]
    mask    = sub_mask.unsqueeze(-1).float()
    pooled  = (sub_emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    empty   = (sub_mask.sum(1) == 0)
    if empty.any():
        pooled[empty] = bert_emb[bert_tok.unk_token_id]
    E[start:start + len(ids)] = pooled
    if start % 5120 == 0:
        L(f"  batch at {start}  ({time.time()-t0:.1f}s)")
L(f"built {tuple(E.shape)} in {time.time()-t0:.1f}s")

assert not torch.isnan(E).any()
def vec(w):
    return E[gpt2_tok.encode(" " + w)[0]]
L("cos(cat,dog)        =", round(float(F.cosine_similarity(vec("cat"), vec("dog"), 0)), 3))
L("cos(cat,carburetor) =", round(float(F.cosine_similarity(vec("cat"), vec("carburetor"), 0)), 3))

os.makedirs("results/exp7_shared", exist_ok=True)
torch.save(E, "results/exp7_shared/shared_emb_gpt2.pt")
L("saved cache")
L("VERIFIED")
log.close()
