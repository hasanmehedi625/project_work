#!/bin/bash
# Fits and saves only the final ensemble (skips cross-validation) — use this
# once CV is already done and you just want the trained model to score on
# unseen data. Same dual-mode pattern as the other scripts:
#   bash run_final_model.sh    -> submits a job and exits
#   sbatch run_final_model.sh  -> runs directly inside the allocation

DATA_CSV="/home/mdmehedi.hasan/Power Outage Modeling/15events_post_event_dates_clean.csv"
OUT_DIR="/home/mdmehedi.hasan/Power Outage Modeling/baseline_model_final_stage1"

set -e
ENV_NAME=outage_env
SCRIPT_NAME=outage_model_stage1.py

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    WORK_DIR="$SLURM_SUBMIT_DIR"
else
    WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
[ -f "$WORK_DIR/$SCRIPT_NAME" ] || SCRIPT_NAME=outage_model.py

module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

if [ -n "$SLURM_JOB_ID" ]; then
    cd "$WORK_DIR"
    echo "running inside SLURM job $SLURM_JOB_ID, using $SLURM_CPUS_PER_TASK cpus"
    python "$SCRIPT_NAME" --data "$DATA_CSV" --out "$OUT_DIR" --seed 0 --final-only
    echo "finished: $(date)"
    exit 0
fi

ASSOC=$(sacctmgr show associations user="$USER" format=account,partition -n -P 2>/dev/null | head -n 1)
ACCOUNT=$(echo "$ASSOC" | cut -d'|' -f1)
PARTITION=$(echo "$ASSOC" | cut -d'|' -f2)
PART_LINE=""
[ -n "$PARTITION" ] && PART_LINE="#SBATCH --partition=$PARTITION"

JOB_SCRIPT="$WORK_DIR/.final_model_job.sbatch"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=final_outage_model
#SBATCH --account=$ACCOUNT
$PART_LINE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

bash "$WORK_DIR/$(basename "${BASH_SOURCE[0]}")"
EOF

echo "submitting final-model job (account=$ACCOUNT, partition=${PARTITION:-default})..."
sbatch "$JOB_SCRIPT"