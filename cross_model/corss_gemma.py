import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)
from TokeniserMap import TokenizerMap
HF_TOKEN = "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"
SMALL_MODEL_ID = "google/gemma-2-2b-it"   # pruner
MAIN_MODEL_ID  = "CohereLabs/aya-expanse-8b"   # generator

# optional 4-bit NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ────────────────────────────────────────────────────────────
# 1.  Gemma-style “weights-only” attention
# ────────────────────────────────────────────────────────────
class GemmaAttentionWeightsOnly(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.q_proj, self.k_proj = original_attn.q_proj, original_attn.k_proj
        self.head_dim   = original_attn.head_dim
        self.num_q      = self.q_proj.out_features // self.head_dim
        self.num_kv     = self.k_proj.out_features // self.head_dim
        self.expand     = self.num_q // self.num_kv
        self.scaling    = getattr(original_attn, "scaling", self.head_dim ** -0.5)
        self.softcap    = getattr(original_attn, "attn_logit_softcapping", None)  # ≈50

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _rope(self, q, k, cos, sin):
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return (
            q * cos + self._rotate_half(q) * sin,
            k * cos + self._rotate_half(k) * sin,
        )

    def forward(self, hidden, rotary, attn_mask):
        B, L, _ = hidden.shape

        q = self.q_proj(hidden).view(B, L, self.num_q,  self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(B, L, self.num_kv, self.head_dim).transpose(1, 2)
        q, k = self._rope(q, k, rotary[0], rotary[1])

        k = k.repeat_interleave(self.expand, dim=1)                       # GQA
        scores = (q @ k.transpose(-1, -2)) * self.scaling                 # [B,h,L,L]

        if self.softcap is not None:
            scores = torch.tanh(scores / self.softcap) * self.softcap     # Gemma trick

        scores = scores + attn_mask
        return F.softmax(scores, dim=-1)                                  # weights


# ────────────────────────────────────────────────────────────
# 2.  TokenPruner
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
        self.self_attention    = GemmaAttentionWeightsOnly(
            small.model.layers[0].self_attn
        )
        del small
        self.compression_ratio = compression_ratio

    def forward(self, input_ids, attention_mask=None):
        embs = self.embeddings(input_ids)                 # [B,L,D]
        B, L, _ = embs.shape

        pos  = torch.arange(L, device=embs.device).unsqueeze(0)
        rotary = self.rotary_embeddings(embs, position_ids=pos)

        # causal attn mask  (0 on/below diagonal, –inf above)
        m = torch.triu(
            torch.full((L, L), -float("inf"), device=embs.device), 1
        ).unsqueeze(0).unsqueeze(0)

        attn = self.self_attention(embs, rotary, m)       # [B,h,L,L]
        importance = attn.mean(dim=1).mean(dim=1)         # [B,L]

        l = int(L * self.compression_ratio)
        k = max(l, 2)
        _, topk = torch.topk(importance, k, sorted=False, dim=-1)
        topk = torch.sort(topk, dim=-1)[0]

        # keep final token
        need_append = (topk[:, -1] != L - 1)
        last = torch.tensor(L - 1, device=topk.device)
        if need_append:
            last_col = torch.full((B, 1), L - 1, device=topk.device)
            topk = torch.cat([topk, last_col], dim=1)

        pruned_ids = torch.gather(input_ids, 1, topk)
        return pruned_ids, topk


# ────────────────────────────────────────────────────────────
# 3.  GemmaPrunedModel  (same API as your LlamaPrunedModel)
# ────────────────────────────────────────────────────────────
class CrossPrunerModel(nn.Module):
    def __init__(self, main_id, small_id, compression_ratio):
        super().__init__()
        self.main_model = AutoModelForCausalLM.from_pretrained(
            main_id,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # quantization_config=bnb_config,   # optional
            trust_remote_code=True,
        )
        self.main_tokenizer   = AutoTokenizer.from_pretrained(main_id,  token=HF_TOKEN,
                                                              trust_remote_code=True)
        self.pruner_tokenizer = AutoTokenizer.from_pretrained(small_id, token=HF_TOKEN,
                                                              trust_remote_code=True)

        self.device = self.main_model.device
        self.embeddings = self.main_model.get_input_embeddings()
        self.token_pruner = TokenPruner(small_id, compression_ratio)
        # freeze generator weights
        self.main_model.requires_grad_(False)
        self.tokenizer_map = TokenizerMap(small_id, main_id)
        self.compression_ratio = compression_ratio

    # — helper that runs the pruner —
    def post_tokenizer(self, input_ids,atention_mask=None):
        pruned_tokens_ids,_ = self.token_pruner(input_ids.to("cuda:0"),atention_mask)
        pruned_tokens_ids = pruned_tokens_ids.tolist()
        bos_id = self.pruner_tokenizer.bos_token_id
        for i in range(len(pruned_tokens_ids)):
            if pruned_tokens_ids[i][0] == bos_id:
                pruned_tokens_ids[i] = pruned_tokens_ids[i][1:]
        pruned_tokens = self.pruner_tokenizer.batch_decode(pruned_tokens_ids, skip_special_tokens=False)
        for i in range(len(pruned_tokens)):
            pruned_tokens[i] = self.tokenizer_map.map_string(pruned_tokens[i])
        return pruned_tokens
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if self.compression_ratio == 1.0:
            return self.main_model.forward(input_ids,attention_mask=attention_mask, **kwargs)
        pruned_tokens = self.post_tokenizer(input_ids, attention_mask)
        output = self.main_model.forward(**self.main_tokenizer(pruned_tokens,return_tensors="pt").to(self.device), **kwargs)

        return output
    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if self.compression_ratio == 1.0:
            return self.main_model.forward(input_ids,attention_mask=attention_mask, **kwargs)
        pruned_tokens= self.post_tokenizer(input_ids, attention_mask)
        output = self.main_model.generate(**self.main_tokenizer(pruned_tokens,return_tensors="pt").to(self.device), **kwargs)
        return output
