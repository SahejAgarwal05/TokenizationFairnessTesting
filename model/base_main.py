import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from model.gemma2_prunner import TokenPruner as Gemma2TokenPruner
from model.llama3_pruner import TokenPruner as Llama3TokenPruner
# import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
class PrunedModel(nn.Module):
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
        self.pruner_tokenizer = AutoTokenizer.from_pretrained(
            small_model_id,
            trust_remote_code=True,
            token=token,
        )
        self.config = AutoConfig.from_pretrained(main_model_id)
        self.device = self.main_model.device
        self.tie_weights = lambda: self
        self.embeddings = self.main_model.get_input_embeddings()
        self.cross_model_flag = False
        if "gemma" in small_model_id:
            self.token_pruner = Gemma2TokenPruner(
                small_model_id, compression_ratio, device="cuda:0"
            )
            if "gemma" in main_model_id:
                self.cross_model_flag = False
        elif "llama" in small_model_id:
            self.token_pruner = Llama3TokenPruner(
                small_model_id, compression_ratio, device="cuda:0"
            )
            if "llama" in main_model_id:
                self.cross_model_flag = False
        self.compression_ratio = compression_ratio
        # Freeze main model
        for param in self.main_model.parameters():
            param.requires_grad = False
        self.embeddings.requires_grad = False
    def post_tokenizer(self, input_ids,atention_mask=None):
        pruned_tokens_ids,_ = self.token_pruner(input_ids.to("cuda:0"),atention_mask)
        pruned_tokens = self.pruner_tokenizer.batch_decode(pruned_tokens_ids, skip_special_tokens=False)
        return pruned_tokens
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if self.compression_ratio == 1.0:
            return self.main_model.forward(input_ids,attention_mask=attention_mask, **kwargs)
        pruned_tokens = self.post_tokenizer(input_ids, attention_mask)
        output = self.main_model.forward(**self.main_tokenizer(pruned_tokens,return_tensors="pt",add_special_tokens=False).to(self.device), **kwargs)
        return output
    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        if self.compression_ratio == 1.0:
            return self.main_model.generate(input_ids,attention_mask=attention_mask, **kwargs)
        pruned_tokens= self.post_tokenizer(input_ids, attention_mask)
        output = self.main_model.generate(**self.main_tokenizer(pruned_tokens,return_tensors="pt",add_special_tokens=False).to(self.device), **kwargs)
        return output