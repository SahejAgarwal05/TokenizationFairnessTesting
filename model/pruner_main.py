from transformers import  AutoTokenizer
from model.gemma2_prunner import TokenPruner as Gemma2TokenPruner
from model.llama3_pruner import TokenPruner as Llama3TokenPruner
from model.across_family_map import (
    convert_aya_expanse_to_gemma2,
    convert_aya_expanse_to_llama3,
    convert_gemma2_to_llama3,
    convert_gemma2_to_aya_expanse,
    convert_llama3_to_gemma2,
    convert_llama3_to_aya_expanse,
)
class Pruner:
    def __init__(self, small_model_id, main_model_id, compression_ratio,device="cuda:0"):
        self.main_model_id = main_model_id
        self.small_model_id = small_model_id
        self.device = device
        self.pruner_tokenizer = AutoTokenizer.from_pretrained(
            small_model_id,
            trust_remote_code=True,
        )
        self.main_tokenizer = AutoTokenizer.from_pretrained(
            main_model_id,
            trust_remote_code=True,
        )
        self.compression_ratio = compression_ratio
        self.across_family_flag = False
        self.across_family_forward_fn = lambda x: x
        self.across_family_backward_fn = lambda x: x
        if "gemma" in small_model_id:
            self.token_pruner = Gemma2TokenPruner(
                small_model_id, compression_ratio, device=self.device
            )
            if "llama" in main_model_id:
                self.across_family_backward_fn = lambda x: convert_gemma2_to_llama3(
                    x
                )
                self.across_family_forward_fn = lambda x: convert_llama3_to_gemma2(x)
                self.across_family_flag = True
            if "aya-expanse" in main_model_id:
                self.across_family_backward_fn = (
                    lambda x: convert_gemma2_to_aya_expanse(x)
                )
                self.across_family_forward_fn = lambda x: convert_aya_expanse_to_gemma2(
                    x
                )
                self.across_family_flag = True
        elif "llama" in small_model_id:
            self.token_pruner = Llama3TokenPruner(
                small_model_id, compression_ratio, device=self.device
            )
            if "gemma" in main_model_id:
                self.across_family_backward_fn = lambda x: convert_llama3_to_gemma2(
                    x
                )
                self.across_family_forward_fn = lambda x: convert_gemma2_to_llama3(x)
                self.across_family_flag = True
            if "aya-expanse" in main_model_id:
                self.across_family_backward_fn = (
                    lambda x: convert_llama3_to_aya_expanse(x)
                )
                self.across_family_forward_fn = lambda x: convert_aya_expanse_to_llama3(
                    x
                )
                self.across_family_flag = True
    def __call__(self, input_ids,attention_mask=None,input_text_list=False):
        if self.across_family_flag:
            if input_text_list:
                org_tokens = input_ids
            else:
                org_tokens = self.main_tokenizer.batch_decode(
                    input_ids, skip_special_tokens=False
                )
            new_tokens = [self.across_family_forward_fn(o_t) for o_t in org_tokens]
            new_inputs = self.pruner_tokenizer(
                new_tokens, return_tensors="pt", add_special_tokens=False
            )
            input_ids = new_inputs["input_ids"]
            attention_mask = new_inputs["attention_mask"]
        pruned_tokens_ids, _ = self.token_pruner(input_ids.to("cuda:0"), attention_mask)
        pruned_tokens = self.pruner_tokenizer.batch_decode(
            pruned_tokens_ids, skip_special_tokens=False
        )
        if self.across_family_flag:
            pruned_tokens = [self.across_family_backward_fn(p_t) for p_t in pruned_tokens]
        return pruned_tokens
