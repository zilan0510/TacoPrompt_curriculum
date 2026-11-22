#!/bin/bash
#SBATCH --job-name=tp_dump_generic
#SBATCH --partition=RTX4090,RTX3090
#SBATCH --exclude=main-1,main-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/out/dump_%x_%j.out
#SBATCH --error=slurm/out/dump_%x_%j.err

# 通用 dump：CFG=<config> TAG=<tag> SHARD=<i> NUM=<n> MODEL_DIR=<dir with model_best.pth>
SHARD=${1:-0}
NUM=${2:-1}

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tacopromptt
cd ~/landou0510/project/TacoPrompt/TacoPrompt
export WANDB_MODE=offline
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

: "${CFG:?need CFG}"; : "${TAG:?need TAG}"; : "${MODEL_DIR:?need MODEL_DIR}"
echo "job start: $(date) cfg=${CFG} tag=${TAG} shard=${SHARD}/${NUM} model=${MODEL_DIR}"
set -e
python analysis/dump_hidden_rank.py \
  -c "${CFG}" \
  -m "${MODEL_DIR}/model_best.pth" \
  -o "analysis/results/${TAG}_top10_shard${SHARD}.json" \
  --shard ${SHARD} --num_shards ${NUM} --verify
echo "job end: $(date)"
