export CUDA_VISIBLE_DEVICES=0,1
python lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=CohereForAI/aya-expanse-8b --compression_ratio=1.0 --task=global_mmlu_lite
export CUDA_VISIBLE_DEVICES=0,1
python lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=CohereForAI/aya-expanse-8b --compression_ratio=0.9 --task=global_mmlu_lite
export CUDA_VISIBLE_DEVICES=0,1
python lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=CohereForAI/aya-expanse-8b --compression_ratio=0.8 --task=global_mmlu_lite
export CUDA_VISIBLE_DEVICES=0,1
python lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=CohereForAI/aya-expanse-8b --compression_ratio=0.7 --task=global_mmlu_lite