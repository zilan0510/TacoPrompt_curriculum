#!/bin/bash
#SBATCH --job-name=tp_dump_verb
#SBATCH --partition=RTX4090
#SBATCH --exclude=main-1,main-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/out/dump_verb_%j.out
#SBATCH --error=slurm/out/dump_verb_%j.err

# 用法: MODEL_DIR=<含 model_best.pth 的目录> sbatch slurm/run_dump_verb.slurm <shard> <num_shards>
SHARD=${1:-0}
NUM=${2:-1}

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tacopromptt
cd ~/landou0510/project/TacoPrompt/TacoPrompt

export WANDB_MODE=offline
export WANDB_SILENT=true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "job start: $(date)  shard=${SHARD}/${NUM}  model=${MODEL_DIR}"
python analysis/dump_hidden_rank.py \
  -c config_files/SemEval-V/config.test.hidden.json \
  -m "${MODEL_DIR}/model_best.pth" \
  -o analysis/results/verb_top10_shard${SHARD}.json \
  --shard ${SHARD} --num_shards ${NUM} --verify
echo "job end: $(date)"
