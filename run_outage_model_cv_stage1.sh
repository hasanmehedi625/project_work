#!/bin/bash
# Self-configuring runner for outage_model.py on Kamiak.
#
# Works either way you invoke it:
#   bash run_outage_model_stage1.sh     -> looks up account/partition, builds
#                                          the conda env if needed, submits a
#                                          real job, and exits immediately.
#   sbatch run_outage_model_stage1.sh   -> detects it's already running inside
#                                          a SLURM allocation and just runs
#                                          the python job directly, using the
#                                          resources that job already has.
# Either way the compute step is identical; only how it gets scheduled differs.

set -e

ENV_NAME=outage_env
SCRIPT_NAME=outage_model_stage1.py

# Where the real files live: prefer where you launched from, not wherever
# SLURM happens to be executing this copy of the script from.
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    WORK_DIR="$SLURM_SUBMIT_DIR"
else
    WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
[ -f "$WORK_DIR/$SCRIPT_NAME" ] || SCRIPT_NAME=outage_model.py

# --- conda env: find it, or build it, either invocation mode ---
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "conda env '$ENV_NAME' not found — creating it (this can take several minutes)"
    conda create -n "$ENV_NAME" python=3.10 -y
    conda activate "$ENV_NAME"
    pip install --quiet torch pandas scikit-learn matplotlib
else
    conda activate "$ENV_NAME"
fi

if ! python -c "import torch, pandas, sklearn, matplotlib" 2>/dev/null; then
    echo "'$ENV_NAME' is missing required packages — installing them now"
    pip install --quiet torch pandas scikit-learn matplotlib
fi

# --- mode A: already inside a SLURM job (invoked via sbatch) -> just run ---
if [ -n "$SLURM_JOB_ID" ]; then
    cd "$WORK_DIR"
    echo "running inside SLURM job $SLURM_JOB_ID, using $SLURM_CPUS_PER_TASK cpus"
    python "$SCRIPT_NAME" \
        --data "$WORK_DIR/15events_post_event_dates_clean.csv" \
        --out  "$WORK_DIR/baseline_cv_2" \
        --seed 0
    echo "finished: $(date)"
    exit 0
fi

# --- mode B: plain bash invocation -> look up account/partition and submit ---
ASSOC=$(sacctmgr show associations user="$USER" format=account,partition -n -P 2>/dev/null | grep -v '|$' | head -n 1)
if [ -z "$ASSOC" ]; then
    # fall back to an account with no partition restriction listed
    ASSOC=$(sacctmgr show associations user="$USER" format=account,partition -n -P 2>/dev/null | head -n 1)
fi
if [ -z "$ASSOC" ]; then
    echo "Could not find any SLURM account for $USER via sacctmgr. Contact HPC support"
    echo "to confirm you have an allocation before running this again."
    exit 1
fi
ACCOUNT=$(echo "$ASSOC" | cut -d'|' -f1)
PARTITION=$(echo "$ASSOC" | cut -d'|' -f2)

if [ -z "$PARTITION" ]; then
    echo "Using account='$ACCOUNT' with no specific partition (letting SLURM pick the default)."
    PART_LINE=""
else
    echo "Using account='$ACCOUNT' partition='$PARTITION'"
    PART_LINE="#SBATCH --partition=$PARTITION"
fi

# Job script goes in $WORK_DIR (your own directory, always writable by you) —
# never in a SLURM spool dir, which is what caused the "Permission denied".
JOB_SCRIPT="$WORK_DIR/.outage_model_job.sbatch"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=outage_model
#SBATCH --account=$ACCOUNT
$PART_LINE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Re-invoke this same wrapper; it will detect SLURM_JOB_ID and take mode A above.
bash "$WORK_DIR/$(basename "${BASH_SOURCE[0]}")"
EOF

echo "submitting job..."
sbatch "$JOB_SCRIPT"