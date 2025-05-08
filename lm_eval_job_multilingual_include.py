import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
        "include_base_44_albanian",
        "include_base_44_arabic",
        "include_base_44_armenia",
        "include_base_44_azerbaijani",
        "include_base_44_basque",
        "include_base_44_belarusian",
        "include_base_44_bengali",
        "include_base_44_bulgarian",
        "include_base_44_chinese",
        "include_base_44_croatian",
        "include_base_44_dutch",
        "include_base_44_estonian",
        "include_base_44_finnish",
        "include_base_44_french",
        "include_base_44_georgian",
        "include_base_44_german",
        "include_base_44_greek",
        "include_base_44_hebrew",
        "include_base_44_hindi",
        "include_base_44_hungarian",
        "include_base_44_indonesian",
        "include_base_44_italian",
        "include_base_44_japanese",
        "include_base_44_kazakh",
        "include_base_44_korean",
        "include_base_44_lithuanian",
        "include_base_44_malay",
        "include_base_44_malayalam",
        "include_base_44_nepali",
        "include_base_44_north_macedonian",
        "include_base_44_persian",
        "include_base_44_polish",
        "include_base_44_portuguese",
        "include_base_44_russian",
        "include_base_44_serbian",
        "include_base_44_spanish",
        "include_base_44_tagalog",
        "include_base_44_tamil",
        "include_base_44_telugu",
        "include_base_44_turkish",
        "include_base_44_ukrainian",
        "include_base_44_urdu",
        "include_base_44_uzbek",
        "include_base_44_vietnamese",
    ],  # Albanian, Arabic, Armenian, Azerbaijani, Basque, Belarusian, Bengali, Bulgarian, Chinese, Croatian, Dutch, Estonian, Finnish, French, Georgian, German, Greek, Hebrew, Hindi, Hungarian, Indonesia, Italian, Japanese, Kazakh, Korean, Lithuanian, Malay, Malayalam, Nepali, North Macedonian, Persian, Polish, Portuguese, russian, Serbian, Spanish, Tagalog, Tamil, Telugu, Turkish, Ukrainian, Urdu, Uzbek, Vietnamese

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
