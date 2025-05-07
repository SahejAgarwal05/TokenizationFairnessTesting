import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import lm_eval
from lm_eval.utils import setup_logging
from lm_eval.loggers import WandbLogger
from transformers import AutoTokenizer
from model_main import LlamaPrunedModel
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
token = "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"

small_model_id = "meta-llama/Llama-3.2-1B-Instruct"
main_model_id = "meta-llama/Llama-3.1-8B-Instruct"
compression_ratio = 0.9

tokenizer = AutoTokenizer.from_pretrained(main_model_id)

model = LlamaPrunedModel(main_model_id, small_model_id, compression_ratio,token=token)
# initialize logging
task_manager = TaskManager()
setup_logging("DEBUG") # optional, but recommended; or you can set up logging yourself
results = lm_eval.simple_evaluate( # call simple_evaluate
    model=HFLM(pretrained=model, tokenizer=tokenizer),
    tasks=["mgsm_cot_native"],
    #tasks=["mmlu"],
    # num_fewshot=5,
    log_samples=True,
    # batch_size=16,
    task_manager=task_manager,
    apply_chat_template=True,
)

wandb_logger = WandbLogger()
wandb_logger.post_init(results)
wandb_logger.log_eval_result()
wandb_logger.log_eval_samples(results["samples"])  # if log_samples
