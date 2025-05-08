import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import lm_eval
from lm_eval.utils import setup_logging
from lm_eval.loggers import WandbLogger
from transformers import AutoTokenizer
from model_main import LlamaPrunedModel
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
from Gemma2.main_model import GemmaPrunedModel
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--small_model",
    default="meta-llama/Llama-3.2-1B-Instruct",
    type=str,
)
parser.add_argument(
    "--main_model",
    default="meta-llama/Llama-3.1-8B-Instruct",
    type=str,
)
parser.add_argument(
    "--compression_ratio",
    default=0.9,
    type=float,
)
args = parser.parse_args()

token = "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"

small_model_id = args.small_model
main_model_id = args.main_model
compression_ratio = args.compression_ratio

tokenizer = AutoTokenizer.from_pretrained(main_model_id)

if "Gemma" in main_model_id:
    model = GemmaPrunedModel(main_model_id, small_model_id, compression_ratio)
else:
    model = LlamaPrunedModel(main_model_id, small_model_id, compression_ratio,token=token)
# initialize logging
task_manager = TaskManager()
setup_logging("DEBUG")  # optional, but recommended; or you can set up logging yourself
results = lm_eval.simple_evaluate(  # call simple_evaluate
    model=HFLM(pretrained=model, tokenizer=tokenizer),
    # tasks=["mmlu_it_llama","mmlu_fr_llama","mmlu_pt_llama","mmlu_th_llama","mmlu_hi_llama","mmlu_de_llama","mmlu_es_llama"],
    # tasks=["mmlu"],
    tasks=[
        "global_mmlu_ar",
        "global_mmlu_bn",
        "global_mmlu_de",
        "global_mmlu_en",
        "global_mmlu_fr",
        "global_mmlu_hi",
        "global_mmlu_id",
        "global_mmlu_it",
        "global_mmlu_ja",
        "global_mmlu_ko",
        "global_mmlu_pt",
        "global_mmlu_es",
        "global_mmlu_sw",
        "global_mmlu_yo",
        "global_mmlu_zh",
    ],  #global_mmlu_lite

    num_fewshot=5,
    log_samples=True,
    # batch_size=16,
    task_manager=task_manager,
    apply_chat_template=True,
    fewshot_as_multiturn=True,
)

wandb_logger = WandbLogger()
wandb_logger.post_init(results)
wandb_logger.log_eval_result()
wandb_logger.log_eval_samples(results["samples"])  # if log_samples
print("small_model_id: " + small_model_id + " main_model_id: " + main_model_id + " compression_ratio: " + str(compression_ratio) + "tasks: Global MMLU")
