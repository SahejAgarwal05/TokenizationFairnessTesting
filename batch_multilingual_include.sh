export CUDA_VISIBLE_DEVICES=2
python lm_eval_job_multilingual_include.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=meta-llama/Llama-3.1-8B-Instruct --compression_ratio=1.0  --task=include
export CUDA_VISIBLE_DEVICES=2
python lm_eval_job_multilingual_include.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=meta-llama/Llama-3.1-8B-Instruct --compression_ratio=0.9  --task=include
export CUDA_VISIBLE_DEVICES=2
python lm_eval_job_multilingual_include.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=meta-llama/Llama-3.1-8B-Instruct --compression_ratio=0.8  --task=include
export CUDA_VISIBLE_DEVICES=2
python lm_eval_job_multilingual_include.py --small_model=meta-llama/Llama-3.2-1B-Instruct --main_model=meta-llama/Llama-3.1-8B-Instruct --compression_ratio=0.7  --task=include