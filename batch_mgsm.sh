export CUDA_VISIBLE_DEVICES=3
python lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=meta-llama/Llama-3.1-8B-Instruct --compression_ratio=1.0 --task=mgsm
export CUDA_VISIBLE_DEVICES=3
python lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=meta-llama/Llama-3.1-8B-Instruct --compression_ratio=0.9 --task=mgsm
export CUDA_VISIBLE_DEVICES=3
python lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=meta-llama/Llama-3.1-8B-Instruct --compression_ratio=0.8 --task=mgsm
export CUDA_VISIBLE_DEVICES=3
python lm_eval_job.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=meta-llama/Llama-3.1-8B-Instruct --compression_ratio=0.7 --task=mgsm