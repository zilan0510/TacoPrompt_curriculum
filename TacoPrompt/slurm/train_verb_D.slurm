#!/bin/bash
#SBATCH --job-name=tp_verb_D
#SBATCH --partition=RTX4090,RTX3090,RTX2080
#SBATCH --exclude=main-1,main-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm/out/train_verb_D_%j.out
#SBATCH --error=slurm/out/train_verb_D_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tacopromptt
cd ~/landou0510/project/TacoPrompt/TacoPrompt
export WANDB_MODE=offline
export WANDB_SILENT=true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "job start: $(date)  cfg=config_files/SemEval-V/config.test.hidden.hardD.json"
set -e
python train.py --config config_files/SemEval-V/config.test.hidden.hardD.json
echo "job end: $(date)"
