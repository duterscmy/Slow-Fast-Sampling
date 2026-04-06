#!/bin/bash
#SBATCH --job-name="soar_eval"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # 请求2块GPU
#SBATCH --partition=a100
#SBATCH --time=12:00:00
#SBATCH -o slurm.%j.%N.out
#SBATCH -e slurm.%j.%N.err

### 激活conda环境
source ~/.bashrc # 你的环境名
conda activate ttrl_env
cd /mnt/fast/nobackup/scratch4weeks/mc03002/Slow-Fast-Sampling/slow-fast-sampling

export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=0
export CUDA_VISIBLE_DEVICES=0

accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k_cot_zeroshot --batch_size 1 \
--model_args "pretrained=/mnt/fast/nobackup/scratch4weeks/mc03002/models/LLaDA-8B-Instruct,parallelize=False,backend="causal",mc_num=128,is_feature_cache=False,is_cfg_cache=False" \
--gen_kwargs "block_length=32,gen_length=256,cfg_scale=0.0 "  \
--num_fewshot 0  \
--output_path ./gsm8k_log \
--log_samples \

# accelerate launch --config_file eval_config.yaml evaluation_script.py -m lm_eval --model LLADA --tasks gsm8k --batch_size 1 \
# --model_args "pretrained=GSAI-ML/LLaDA-8B-Base,parallelize=False,prompt_interval_steps=15,gen_interval_steps=1,transfer_ratio=0.0,is_feature_cache=True,is_cfg_cache=False" \
# --gen_kwargs "block_length=256,gen_length=256,cfg_scale=0.0 "  \
# --num_fewshot 4  \
# --output_path ./gsm8k_log \
# --log_samples \