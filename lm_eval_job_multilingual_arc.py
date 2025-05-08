import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

if "gemma" in main_model_id:
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
        "arc_ar",
        "arc_bn",
        "arc_ca",
        "arc_da",
        "arc_de",
        "arc_es",
        "arc_eu",
        "arc_fr",
        "arc_gu",
        "arc_hi",
        "arc_hr",
        "arc_hu",
        "arc_hy",
        "arc_id",
        "arc_it",
        "arc_kn",
        "arc_ml",
        "arc_mr",
        "arc_ne",
        "arc_nl",
        "arc_pt",
        "arc_ro",
        "arc_ru",
        "arc_sk",
        "arc_sr",
        "arc_sv",
        "arc_ta",
        "arc_te",
        "arc_uk",
        "arc_vi",
        "arc_zh",
    ],  #arc_{ar,bn,ca,da,de,es,eu,fr,gu,hi,hr,hu,hy,id,it,kn,ml,mr,ne,nl,pt,ro,ru,sk,sr,sv,ta,te,uk,vi,zh}

    num_fewshot=5,
    log_samples=True,
    # batch_size=16,
    task_manager=task_manager,
    apply_chat_template=True,
    # fewshot_as_multiturn=True,
)

wandb_logger = WandbLogger()
wandb_logger.post_init(results)
wandb_logger.log_eval_result()
wandb_logger.log_eval_samples(results["samples"])  # if log_samples
print("small_model_id: " + small_model_id + " main_model_id: " + main_model_id + " compression_ratio: " + str(compression_ratio) + "tasks: Global MMLU")
