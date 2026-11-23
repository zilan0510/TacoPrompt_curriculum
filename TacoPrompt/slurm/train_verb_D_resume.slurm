#!/bin/bash
#SBATCH --job-name=tp_verb_D_rs
#SBATCH --partition=RTX4090,RTX3090
#SBATCH --exclude=main-1,main-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/out/train_verb_D_resume_%j.out
#SBATCH --error=slurm/out/train_verb_D_resume_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tacopromptt
cd ~/landou0510/project/TacoPrompt/TacoPrompt
export WANDB_MODE=offline
export WANDB_SILENT=true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 133627 在 epoch40 首次验证时 OOM（2080Ti 10.57G，test_batch_size=775 需 11.28G）。
# 与 baseline 续训完全同口径：从 epoch40 续训 + --tbs 400 + expandable_segments。
echo "job start: $(date)"
set -e
python train.py --config config_files/SemEval-V/config.test.hidden.hardD.json \
  -r data/saved/wordnet_verb/TacoPromptHidden/1120_211631/models/trial1/checkpoint-epoch40.pth \
  --tbs 400 --suffix verb_D_resume
echo "job end: $(date)"
