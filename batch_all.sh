python lm_eval_job.py --small_model=google/gemma-2-2b-it --main_model=google/gemma-2-27b-it --compression_ratio=1.0 --task=global_mmlu_lite
python lm_eval_job.py --small_model=google/gemma-2-2b-it --main_model=google/gemma-2-27b-it --compression_ratio=0.9 --task=global_mmlu_lite
python lm_eval_job.py --small_model=google/gemma-2-2b-it --main_model=google/gemma-2-27b-it --compression_ratio=0.8 --task=global_mmlu_lite
python lm_eval_job.py --small_model=google/gemma-2-2b-it --main_model=google/gemma-2-27b-it --compression_ratio=1.0 --task=mgsm
python lm_eval_job.py --small_model=google/gemma-2-2b-it --main_model=google/gemma-2-27b-it --compression_ratio=0.9 --task=mgsm
python lm_eval_job.py --small_model=google/gemma-2-2b-it --main_model=google/gemma-2-27b-it --compression_ratio=0.8 --task=mgsm
python lm_eval_job.py --small_model=google/gemma-2-2b-it --main_model=google/gemma-2-27b-it --compression_ratio=1.0 --task=arc
python lm_eval_job.py --small_model=google/gemma-2-2b-it --main_model=google/gemma-2-27b-it --compression_ratio=0.9 --task=arc
python lm_eval_job.py --small_model=google/gemma-2-2b-it --main_model=google/gemma-2-27b-it --compression_ratio=0.8 --task=arc
python lm_eval_job.py --small_model=google/gemma-2-2b-it --main_model=google/gemma-2-27b-it --compression_ratio=0.7 --task=arcython lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=meta-llama/Llama-3.1-8B-Instruct --compression_ratio=0.7 --task=mgsmython lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=meta-llama/Llama-3.1-8B-Instruct --compression_ratio=0.7 --task=global_mmlu_lite