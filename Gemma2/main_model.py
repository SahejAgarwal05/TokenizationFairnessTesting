# ────────────────────────────────────────────────────────────
# Gemma-pruning pipeline
#   – minimal, self-contained, no dropout –
# ────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

HF_TOKEN        = "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"
SMALL_MODEL_ID  = "google/gemma-2-2b-it"   # pruner
MAIN_MODEL_ID   = "google/gemma-2-9b-it"   # generator

# Optional 4-bit quant
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# ────────────────────────────────────────────────────────────
# 1.  “Weights-only” attention for Gemma
# ────────────────────────────────────────────────────────────
class GemmaAttentionWeightsOnly(nn.Module):
    """
    Returns **only** the attention weights (no value mat-mul) so we
    can rank token importance cheaply.
    """
    def __init__(self, original_attn):
        super().__init__()
        self.q_proj, self.k_proj = original_attn.q_proj, original_attn.k_proj
        self.head_dim = original_attn.head_dim

        self.num_q  = self.q_proj.out_features // self.head_dim
        self.num_kv = self.k_proj.out_features // self.head_dim
        self.expand = self.num_q // self.num_kv                # GQA

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _rope(self, q, k, cos, sin):
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin,
        )

    def forward(self, hidden, rotary, attn_mask):
        """
        hidden:      [B, L, D]
        rotary:      (cos, sin) from Gemma rotary_emb
        attn_mask:   [1, 1, L, L] with 0 / -inf (causal)
        returns:     attention weights [B, h, L, L]
        """
        B, L, _ = hidden.shape

        q = self.q_proj(hidden).view(B, L, self.num_q,  self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, L, self.num_kv, self.head_dim).transpose(1, 2)
        q, k = self._rope(q, k, rotary[0], rotary[1])

        k = k.repeat_interleave(self.expand, dim=1)               # expand KV heads → Q heads
        scores = (q @ k.transpose(-1, -2)) 

        scores = scores #+ attn_mask.to(scores.dtype)              # causal masking

        return F.softmax(scores, dim=-1)                          # weights


# ────────────────────────────────────────────────────────────
# 2.  Tiny model that picks the top-k tokens
# ────────────────────────────────────────────────────────────
class TokenPruner(nn.Module):
    def __init__(self, model_id, compression_ratio, device="cuda:0"):
        super().__init__()
        small = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.embeddings        = small.get_input_embeddings()
        self.rotary_embeddings = small.model.rotary_emb
        self.self_attention    = GemmaAttentionWeightsOnly(small.model.layers[0].self_attn)
        del small

        self.compression_ratio = compression_ratio

    def forward(self, input_ids, attention_mask=None):
        """
        Returns:
            pruned_ids  – input_ids with only top-k kept  [B, k]
            keep_idx    – indices in the original seq     [B, k]
        """
        embs = self.embeddings(input_ids)                     # [B,L,D]
        B, L, _ = embs.shape

        pos = torch.arange(L, device=embs.device).unsqueeze(0)
        rotary = self.rotary_embeddings(embs, position_ids=pos)

        # causal mask (0 on diag / below, -inf above)
        m = torch.triu(
            torch.full((L, L), -float("inf"), device=embs.device), 1
        ).unsqueeze(0).unsqueeze(0)

        attn_w = self.self_attention(embs, rotary, m)         # [B,h,L,L]
        importance = attn_w.mean(dim=1).mean(dim=1)           # [B,L]

        k = max(int(L * self.compression_ratio), 2)
        _, topk = torch.topk(importance, k, sorted=False, dim=-1)
        topk = torch.sort(topk, dim=-1)[0]                    # keep original order

        # always keep the last token (EOS / BOS continuation)
        need_append = topk[:, -1] != (L - 1)
        if need_append.any():
            last = torch.full((B, 1), L - 1, device=topk.device)
            topk = torch.cat([topk, last], dim=1)

        pruned_ids = torch.gather(input_ids, 1, topk)
        return pruned_ids, topk


# ────────────────────────────────────────────────────────────
# 3.  Main model wrapper (identical API to your Llama version)
# ────────────────────────────────────────────────────────────
class GemmaPrunedModel(nn.Module):
    def __init__(self, main_id, small_id, compression_ratio):
        super().__init__()
        self.main_model = AutoModelForCausalLM.from_pretrained(
            main_id,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # quantization_config=bnb_config,   # <-- enable if desired
            trust_remote_code=True,
        )
        self.main_tokenizer   = AutoTokenizer.from_pretrained(main_id,  token=HF_TOKEN,
                                                              trust_remote_code=True)
        self.pruner_tokenizer = AutoTokenizer.from_pretrained(small_id, token=HF_TOKEN,
                                                              trust_remote_code=True)

        self.token_pruner = TokenPruner(small_id, compression_ratio)
        self.compression_ratio = compression_ratio
        self.device = self.main_model.device
        self.main_model.requires_grad_(False)                 # generator frozen

    # helper: run the pruner, then detokenize on the *pruner* vocab
    def _prune_and_detok(self, input_ids, attention_mask=None):
        pruned_ids, _ = self.token_pruner(input_ids.to(next(self.token_pruner.parameters()).device),
                                          attention_mask)
        # drop leading BOS if present (Gemma 2-BOS style)
        bos_id = self.pruner_tokenizer.bos_token_id
        if (pruned_ids[:, 0] == bos_id).any():
            pruned_ids = pruned_ids[:, 1:]
        return self.pruner_tokenizer.batch_decode(pruned_ids, skip_special_tokens=False)

    # -------- forward / generate wrappers --------
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if self.compression_ratio == 1.0:
            return self.main_model(input_ids, attention_mask=attention_mask, **kwargs)

        pruned_text = self._prune_and_detok(input_ids, attention_mask)
        model_inputs = self.main_tokenizer(pruned_text, return_tensors="pt").to(self.device)
        return self.main_model(**model_inputs, **kwargs)

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if self.compression_ratio == 1.0:
            return self.main_model.generate(input_ids, attention_mask=attention_mask, **kwargs)

        pruned_text = self._prune_and_detok(input_ids, attention_mask)
        model_inputs = self.main_tokenizer(pruned_text, return_tensors="pt").to(self.device)
        return self.main_model.generate(**model_inputs, **kwargs)
