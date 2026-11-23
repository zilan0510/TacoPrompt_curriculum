#!/bin/bash
# ============== SLURM 资源配置 ==============
#SBATCH --job-name=tp_hid_verb
#SBATCH --partition=RTX4090,RTX3090
#SBATCH --exclude=main-1,main-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm/out/train_hidden_verb_%j.out
#SBATCH --error=slurm/out/train_hidden_verb_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tacopromptt
cd ~/landou0510/project/TacoPrompt/TacoPrompt

# wandb 服务器不可达（403），强制离线；HF 走本地缓存避免超时重试
export WANDB_MODE=offline
export WANDB_SILENT=true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "job start: $(date)"
python train.py --config config_files/SemEval-V/config.test.hidden.json --suffix verb_A
echo "job end: $(date)"
