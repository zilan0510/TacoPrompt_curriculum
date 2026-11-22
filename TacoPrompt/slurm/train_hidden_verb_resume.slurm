#!/bin/bash
#SBATCH --job-name=tp_verb_resume
#SBATCH --partition=RTX4090,RTX3090
#SBATCH --exclude=main-1,main-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/out/train_hidden_verb_resume_%j.out
#SBATCH --error=slurm/out/train_hidden_verb_resume_%j.err

set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tacopromptt
cd ~/landou0510/project/TacoPrompt/TacoPrompt

export WANDB_MODE=offline
export WANDB_SILENT=true
export TRANSFORMERS_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# 126656 在 epoch40 的第一次验证（batch 775 → MLM logits 约 12GiB）OOM，
# 这里缩小 test_batch_size（只影响分批，不影响排序结果）并开启可扩展显存段避免碎片 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job start: $(date)"
python train.py \
  --config config_files/SemEval-V/config.test.hidden.json \
  -r data/saved/wordnet_verb/TacoPromptHidden/1108_155917/models/trial1/checkpoint-epoch40.pth \
  --tbs 400
echo "job end: $(date)"
