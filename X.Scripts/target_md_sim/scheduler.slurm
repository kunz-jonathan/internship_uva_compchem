#!/bin/bash
#SBATCH --job-name=md_binder
#SBATCH --output=md_binder_%A_%a.log     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpuRTX-2nd
#SBATCH --gpus-per-node=1
#SBATCH --array=1-6%6


#==========================#GROMACS, configs etc.#==========================#


source /home/spack-user-2025/spack/share/spack/setup-env.sh
spack load gromacs@2024.3/qtc


# no coredumps
ulimit -S -c 0
ulimit -s unlimited
export OMP_NUM_THREADS=$(nproc)


#==========================#SCHEDULER SCRIPT#==========================#

if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then

    mkdir -p "$SLURM_SUBMIT_DIR/jobs"
fi 

# input and output folder
INPUT_FOLDER="$SLURM_SUBMIT_DIR/."
OUTPUT_PARENT="$SLURM_SUBMIT_DIR/jobs"

#worker script
WORKER_SCRIPT="$SLURM_SUBMIT_DIR/../../../X.Scripts/md_sim/worker_script.sh"

# Get all files in an array
FILES=($(ls "$INPUT_FOLDER"/* | sort -V))
NUM_FILES=${#FILES[@]}


# Safety check
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_FILES" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID is out of range (0-$((NUM_FILES-1)))"
    exit 1
fi

echo "=== Starting Job $JOB_ID ==="
echo "Total files: $NUM_FILES"
echo "Processing files assigned to job $JOB_ID"



# files will be split up between arrays
for ((i=SLURM_ARRAY_TASK_ID; i<NUM_FILES; i+=SLURM_ARRAY_TASK_COUNT)); do 

    FILE="${FILES[$i]}"
    BASENAME=$(basename "$FILE")
    PREFIX="${BASENAME%%_*}"
    NEWFOLDER="$OUTPUT_PARENT/job_$PREFIX"
    PYTHON_LOG="$NEWFOLDER/python_log_${SLURM_ARRAY_TASK_ID}.log"

    # ---- skip if folder already exists ----
    if [[ -d "$NEWFOLDER" ]]; then
        echo "[Job $i] $NEWFOLDER already exists — skipping"
        continue
    fi
    # --------------------------------------

    mkdir -p "$NEWFOLDER"
    echo "[Job $i/$NUM_FILES] Copying $FILE → $NEWFOLDER/"
    cp "$FILE" "$NEWFOLDER/"

    echo "[Job $i/$NUM_FILES] Running worker script on $BASENAME"
    bash "$WORKER_SCRIPT" "$NEWFOLDER" > "$PYTHON_LOG" 2>&1

    echo "Finished $BASENAME" 

    cd ../../

done

echo "=== Finished Job $SLURM_ARRAY_TASK_ID ==="




