import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import lm_eval
from lm_eval.utils import setup_logging
from lm_eval.loggers import WandbLogger
from transformers import AutoTokenizer
from model.base_main import PrunedModel
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
import argparse
import task_configs
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
parser.add_argument(
    "--n_shot",
    default=5,
    type=int,
)
parser.add_argument(
    "--task",
    default="global_mmlu_lite",
    type=str,
)
args = parser.parse_args()

token = "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"

small_model_id = args.small_model
main_model_id = args.main_model
compression_ratio = args.compression_ratio
if args.task == "global_mmlu_lite":
    tasks = task_configs.global_mmlu_lite
elif args.task == "include":
    tasks = task_configs.include
elif args.task == "mgsm":
    tasks = task_configs.mgsm
elif args.task == "arc":
    tasks = task_configs.arc
tokenizer = AutoTokenizer.from_pretrained(main_model_id)

model = PrunedModel(main_model_id, small_model_id, compression_ratio,token=token)
# initialize logging
print("small_model_id: " + small_model_id + " main_model_id: " + main_model_id + " compression_ratio: " + str(compression_ratio) + "tasks: " + args.task)
task_manager = TaskManager()
setup_logging("DEBUG")  # optional, but recommended; or you can set up logging yourself
results = lm_eval.simple_evaluate(  # call simple_evaluate
    model=HFLM(pretrained=model, tokenizer=tokenizer),
    tasks=tasks,
    num_fewshot=args.n_shot,
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
print("small_model_id: " + small_model_id + " main_model_id: " + main_model_id + " compression_ratio: " + str(compression_ratio) + "tasks: " + args.task)
import gc, torch
del model
gc.collect()
torch.cuda.empty_cache()