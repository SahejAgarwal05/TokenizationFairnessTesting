export CUDA_VISIBLE_DEVICES=2
python lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=CohereForAI/aya-expanse-8b --compression_ratio=1.0 --task=arc
export CUDA_VISIBLE_DEVICES=2
python lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=CohereForAI/aya-expanse-8b --compression_ratio=0.9 --task=arc
export CUDA_VISIBLE_DEVICES=2
python lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=CohereForAI/aya-expanse-8b --compression_ratio=0.8 --task=arc
export CUDA_VISIBLE_DEVICES=2
python lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=CohereForAI/aya-expanse-8b --compression_ratio=0.7 --task=arc
