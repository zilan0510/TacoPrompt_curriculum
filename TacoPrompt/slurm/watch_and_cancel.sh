#!/bin/bash
# 守护脚本：监视 Verb 续训作业，训练循环一结束（早停 或 跑满 epoch）
#   1) 立刻 scancel，避免 trainer 再跑一遍 ~40h 的冗余全量测试
#   2) 自动为新的 model_best.pth 提交 4 个分片的 Top-10 dump 作业
# 用法: nohup bash slurm/watch_and_cancel.sh <jobid> > /dev/null 2>&1 &
set -u
JOB=${1:?need jobid}
cd ~/landou0510/project/TacoPrompt/TacoPrompt
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tacopromptt

LOG=slurm/out/train_hidden_verb_resume_${JOB}.out
WLOG=slurm/out/watcher_${JOB}.log
echo "[$(date '+%F %T')] watcher start for job $JOB (log=$LOG)" >> "$WLOG"

while squeue -h -j "$JOB" >/dev/null 2>&1 && [ -n "$(squeue -h -j "$JOB" 2>/dev/null)" ]; do
    if [ -f "$LOG" ] && grep -qE "Training stops\.|Testing with best model" "$LOG"; then
        echo "[$(date '+%F %T')] training loop finished, cancelling $JOB to skip redundant full test" >> "$WLOG"
        sleep 10
        scancel "$JOB"
        sleep 20
        MODEL_DIR=$(ls -dt data/saved/wordnet_verb/TacoPromptHidden/*/models/trial1 2>/dev/null | head -1)
        if [ -f "${MODEL_DIR}/model_best.pth" ]; then
            echo "[$(date '+%F %T')] model_best found: ${MODEL_DIR}" >> "$WLOG"
            for s in 0 1 2 3; do
                MODEL_DIR="$MODEL_DIR" sbatch slurm/run_dump_verb.slurm "$s" 4 >> "$WLOG" 2>&1
            done
        else
            echo "[$(date '+%F %T')] WARNING: no model_best.pth under ${MODEL_DIR}" >> "$WLOG"
        fi
        exit 0
    fi
    sleep 60
done
echo "[$(date '+%F %T')] job $JOB left the queue; watcher exit" >> "$WLOG"
