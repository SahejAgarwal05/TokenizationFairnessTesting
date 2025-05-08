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
        "_include_base_44_albanian",
        "_include_base_44_arabic",
        "_include_base_44_armenia",
        "_include_base_44_azerbaijani",
        "_include_base_44_basque",
        "_include_base_44_belarusian",
        "_include_base_44_bengali",
        "_include_base_44_bulgarian",
        "_include_base_44_chinese",
        "_include_base_44_croatian",
        "_include_base_44_dutch",
        "_include_base_44_estonian",
        "_include_base_44_finnish",
        "_include_base_44_french",
        "_include_base_44_georgian",
        "_include_base_44_german",
        "_include_base_44_greek",
        "_include_base_44_hebrew",
        "_include_base_44_hindi",
        "_include_base_44_hungarian",
        "_include_base_44_indonesian",
        "_include_base_44_italian",
        "_include_base_44_japanese",
        "_include_base_44_kazakh",
        "_include_base_44_korean",
        "_include_base_44_lithuanian",
        "_include_base_44_malay",
        "_include_base_44_malayalam",
        "_include_base_44_nepali",
        "_include_base_44_north_macedonian",
        "_include_base_44_persian",
        "_include_base_44_polish",
        "_include_base_44_portuguese",
        "_include_base_44_russian",
        "_include_base_44_serbian",
        "_include_base_44_spanish",
        "_include_base_44_tagalog",
        "_include_base_44_tamil",
        "_include_base_44_telugu",
        "_include_base_44_turkish",
        "_include_base_44_ukrainian",
        "_include_base_44_urdu",
        "_include_base_44_uzbek",
        "_include_base_44_vietnamese",
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
