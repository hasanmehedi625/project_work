#!/bin/bash
# Runs outage_model_stage1_test.py on Kamiak. Same dual-mode pattern as
# run_outage_model_stage1.sh: works with either
#   bash run_outage_model_stage1_test.sh      -> submits a job and exits
#   sbatch run_outage_model_stage1_test.sh    -> runs directly inside the allocation
#
# EDIT THESE THREE PATHS before running:
TEST_CSV="/home/mdmehedi.hasan/Power Outage Modeling/training_data_idalia_fl_2023_post.csv"
MODEL_DIR="/home/mdmehedi.hasan/Power Outage Modeling/baseline_model_final_stage1/final_model"
OUT_DIR="/home/mdmehedi.hasan/Power Outage Modeling/test_scores_idalia_2023"

set -e
ENV_NAME=outage_env

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    WORK_DIR="$SLURM_SUBMIT_DIR"
else
    WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

if [ -n "$SLURM_JOB_ID" ]; then
    cd "$WORK_DIR"
    python outage_model_stage1_test.py --model-dir "$MODEL_DIR" --data "$TEST_CSV" --out "$OUT_DIR"
    echo "finished: $(date)"
    exit 0
fi

ASSOC=$(sacctmgr show associations user="$USER" format=account,partition -n -P 2>/dev/null | head -n 1)
ACCOUNT=$(echo "$ASSOC" | cut -d'|' -f1)
PARTITION=$(echo "$ASSOC" | cut -d'|' -f2)
PART_LINE=""
[ -n "$PARTITION" ] && PART_LINE="#SBATCH --partition=$PARTITION"

JOB_SCRIPT="$WORK_DIR/.evaluate_job.sbatch"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=evaluate_outage
#SBATCH --account=$ACCOUNT
$PART_LINE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

bash "$WORK_DIR/$(basename "${BASH_SOURCE[0]}")"
EOF

echo "submitting evaluation job (account=$ACCOUNT, partition=${PARTITION:-default})..."
sbatch "$JOB_SCRIPT"