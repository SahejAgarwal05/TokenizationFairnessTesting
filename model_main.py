import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
token = "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"


# ===============================
# Custom Attention Module
# ===============================
class LlamaAttentionWeightsOnly(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.head_dim = original_attn.head_dim

        self.num_query_heads = self.q_proj.out_features // self.head_dim
        self.num_kv_heads = self.k_proj.out_features // self.head_dim
        self.expand_factor = self.num_query_heads // self.num_kv_heads

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self,
        hidden_states,
        rotary_embeddings,
        attn_mask,
        output_attentions=False,
        **kwargs,
    ):
        batch, seq_len, _ = hidden_states.shape

        # Compute Q and K projections.
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)

        # Reshape Q
        q = q.view(batch, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        # Reshape K
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.apply_rotary_pos_emb(
            q, k, rotary_embeddings[0], rotary_embeddings[1]
        )

        # Expand key heads
        k = k.repeat_interleave(self.expand_factor, dim=1)

        # Dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2))

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        return attn_weights


# ===============================
# TokenPruner Module
# ===============================
class TokenPruner(nn.Module):
    def __init__(self, small_model_id, compression_ratio, device="cuda:0"):
        super().__init__()
        # Load the small model
        small_model = AutoModelForCausalLM.from_pretrained(
            small_model_id,
            trust_remote_code=True,
            token=token,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.embeddings = small_model.get_input_embeddings()
        self.rotary_embeddings = small_model.model.rotary_emb

        # Replace the first self-attention layer
        original_attn = small_model.model.layers[0].self_attn
        self.self_attention1 = LlamaAttentionWeightsOnly(original_attn)

        # Delete the rest of the small model to save memory
        del small_model

        self.compression_ratio = compression_ratio

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embeddings(input_ids)
        batch_size, seq_len, hidden_dim = embeddings.shape

        # Rotary embeddings
        pos_ids = torch.arange(seq_len).unsqueeze(0).to(input_ids.device)
        rotary_embeds = self.rotary_embeddings(input_ids, position_ids=pos_ids)


        attention_weights = self.self_attention1(
            embeddings, rotary_embeds, attn_mask=attention_mask
        )

        # Average over heads and query dimension => token importance
        #################################DIFF###########################################
        # importance_scores = attention_weights.mean(dim=1).mean(dim=1) # [1, seq_len]
        importance_scores = attention_weights.mean(dim=1).mean(dim=1) # [seq_len]
        #################################DIFF###########################################

        # Number of tokens to keep
        compressed_length = int(seq_len * self.compression_ratio)
        compressed_length = max(compressed_length, 2)

        # Top-k token indices
        _, topk_ids = torch.topk(
            importance_scores, compressed_length, sorted=False,dim=-1
        ) # [seq_len]
        # Sort to preserve order
        # topk_ids = torch.sort(topk_ids, dim=1)[0] # [1, seq_len]
        topk_ids = torch.sort(topk_ids,dim=-1)[0] # [seq_len]

        if topk_ids[-1][-1] != seq_len - 1:
            topk_ids = torch.cat(
                (topk_ids, torch.tensor([[seq_len - 1]], device=topk_ids.device).long()), dim=1
            )

        # Gather pruned tokens
        pruned_token_ids = input_ids.gather(1, topk_ids)
        return pruned_token_ids, topk_ids


# ===============================
# LlamaPrunedModel Module
# ===============================
class LlamaPrunedModel(nn.Module):
    def __init__(self, main_model_id, small_model_id, compression_ratio,token):
        super().__init__()
        self.main_model = AutoModelForCausalLM.from_pretrained(
            main_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=token,
            # quantization_config=bnb_config
        )
        self.main_tokenizer = AutoTokenizer.from_pretrained(
            main_model_id,
            trust_remote_code=True,
            token=token,
        )
        self.prunner_tokenizer = AutoTokenizer.from_pretrained(
            small_model_id,
            trust_remote_code=True,
            token=token,
        )
        self.config = AutoConfig.from_pretrained(main_model_id)
        self.device = self.main_model.device
        self.tie_weights = lambda: self
        self.embeddings = self.main_model.get_input_embeddings()
        self.token_pruner = TokenPruner(
            small_model_id, compression_ratio, device="cuda:0"
        )
        self.compression_ratio = compression_ratio
        # Freeze main model
        for param in self.main_model.parameters():
            param.requires_grad = False
        self.embeddings.requires_grad = False
    def post_tokenizer(self, input_ids,atention_mask=None):
        pruned_tokens_ids,_ = self.token_pruner(input_ids.to("cuda:0"),atention_mask)
        pruned_tokens = self.prunner_tokenizer.batch_decode(pruned_tokens_ids, skip_special_tokens=False)
        return pruned_tokens
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if self.compression_ratio == 1:
            return self.main_model.forward(input_ids,attention_mask=attention_mask, **kwargs)
        pruned_tokens = self.post_tokenizer(input_ids, attention_mask)
        output = self.main_model.forward(**self.main_tokenizer(pruned_tokens,return_tensors="pt").to(self.device), **kwargs)
        return output
    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if self.compression_ratio == 1:
            return self.main_model.forward(input_ids,attention_mask=attention_mask, **kwargs)
        pruned_tokens= self.post_tokenizer(input_ids, attention_mask)
        output = self.main_model.generate(**self.main_tokenizer(pruned_tokens,return_tensors="pt").to(self.device), **kwargs)
        return output
