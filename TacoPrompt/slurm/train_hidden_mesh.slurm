#!/bin/bash
# ============== SLURM 资源配置 ==============
#SBATCH --job-name=tp_hid_mesh
#SBATCH --partition=RTX4090
#SBATCH --exclude=main-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm/out/train_hidden_mesh_%j.out
#SBATCH --error=slurm/out/train_hidden_mesh_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tacopromptt
cd ~/landou0510/project/TacoPrompt/TacoPrompt

# wandb 服务器不可达（403），强制离线；HF 走本地缓存避免超时重试
export WANDB_MODE=offline
export WANDB_SILENT=true
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "job start: $(date)"
python train.py --config config_files/mesh/config.test.hidden.json --suffix mesh_A
echo "job end: $(date)"
