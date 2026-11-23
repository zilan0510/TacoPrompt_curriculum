#!/bin/bash
# 训练守卫：训练循环一结束（早停或跑满 epoch）就
#   1) scancel 掉作业，跳过 trainer 自带的那遍冗余全量测试（数小时到几十小时）
#   2) 找到最新的 model_best.pth，按分片自动提交 dump_hidden_rank
#
# 用法:
#   nohup bash slurm/watch_and_dump.sh <jobid> <log_path> <dataset_dir> <config> <tag> <shards> &
set -u
JOB=${1:?need jobid}
LOG=${2:?need log path}
DATASET_DIR=${3:?need dataset dir}
CFG=${4:?need config}
TAG=${5:?need tag}
SHARDS=${6:-2}
EXP=${7:-}
SUF=${8:-}

cd ~/landou0510/project/TacoPrompt/TacoPrompt
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tacopromptt

WLOG=slurm/out/watcher_${JOB}.log
echo "[$(date '+%F %T')] watcher start job=$JOB tag=$TAG log=$LOG" >> "$WLOG"

in_queue() { [ -n "$(squeue -h -j "$JOB" -o '%T' 2>/dev/null)" ]; }
job_state() { sacct -j "$JOB" -X -n -o State 2>/dev/null | head -1 | tr -d ' '; }

do_finish() {
    echo "[$(date '+%F %T')] training loop finished -> scancel $JOB" >> "$WLOG"
    sleep 10
    scancel "$JOB" 2>/dev/null || true
    sleep 20
    MODEL_DIR=$(ls -dt "${DATASET_DIR}"/*/models/trial1 2>/dev/null | head -1)
    if [ -n "$SUF" ]; then
        # 优先用 --suffix 生成的目录名精确定位（同名 exp_name 的并发 run 不会互相拿错）
        for d in $(ls -dt "${DATASET_DIR}"/*/models/trial1 2>/dev/null); do
            tsdir=$(basename "$(dirname "$(dirname "$d")")")
            case "$tsdir" in
                *"_${SUF}") MODEL_DIR="$d"; break ;;
            esac
        done
    elif [ -n "$EXP" ]; then
        # 只认 exp_name 匹配的 run 目录，避免多个 run 并行时拿到别人的 model_best
        for d in $(ls -dt "${DATASET_DIR}"/*/models/trial1 2>/dev/null); do
            cfg="$d/../config.json"
            if [ -f "$cfg" ] && python -c "import json,sys; sys.exit(0 if json.load(open('$cfg')).get('exp_name')=='$EXP' else 1)" 2>/dev/null; then
                MODEL_DIR="$d"
                break
            fi
        done
    fi
    if [ -f "${MODEL_DIR}/model_best.pth" ]; then
        echo "[$(date '+%F %T')] model_best: ${MODEL_DIR}" >> "$WLOG"
        for s in $(seq 0 $((SHARDS - 1))); do
            CFG="$CFG" TAG="$TAG" MODEL_DIR="$MODEL_DIR" \
                sbatch --job-name="dump_${TAG}" slurm/run_dump_generic.slurm "$s" "$SHARDS" >> "$WLOG" 2>&1
        done
    else
        echo "[$(date '+%F %T')] WARNING: no model_best.pth under ${MODEL_DIR}" >> "$WLOG"
    fi
    exit 0
}

miss=0
while true; do
    if [ -f "$LOG" ] && grep -qE "Training stops\.|Testing with best model" "$LOG"; then
        do_finish
    fi
    if in_queue; then
        miss=0
        sleep 60
        continue
    fi
    miss=$((miss + 1))
    if [ "$miss" -ge 2 ]; then
        st=$(job_state)
        case "$st" in
            COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|"")
                echo "[$(date '+%F %T')] job $JOB left the queue (state=$st); watcher exit" >> "$WLOG"
                exit 0 ;;
        esac
        miss=0
    fi
    sleep 10
done
