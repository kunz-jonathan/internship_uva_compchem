#!/bin/bash
#SBATCH --job-name=bc_bax_1F16
#SBATCH --output=bc_bax_%A_%a.log
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --partition=gpuRTX-4th,gpuRTX-3rd,gpuRTX-2nd,gpuRTX-1st
#SBATCH --gres=gpu:1
#SBATCH --mem 50gb
#SBATCH --time 72:00:00
#SBATCH --array=1-11%11


# installation dependent variables
BindCraft_dir = 'BindCraft_uva_internship'
conda_env_name = 'BindCraft_modified'


#  input
SETTINGS="/home/kunzj/internship_uva_compchem/X.Scripts/BindCraft/settings_bindcraft.json"
FILTERS="/home/kunzj/${BindCraft_dir}/settings_filters/peptide_relaxed_filters.json"
ADVANCED="/home/kunzj/${BindCraft_dir}/settings_advanced/peptide_3stage_multimer.json"


# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/${conda_env_name}
export LD_LIBRARY_PATH=${CONDA_BASE}/lib


# Create log dir for Python logs 
mkdir -p "slurm_output_${SLURM_JOB_NAME}"
cd "slurm_output_${SLURM_JOB_NAME}"
PYTHON_LOG="slurm_output_${SLURM_JOB_NAME}/python_log_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

# log information of array
if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then {
    echo "====================================="
    echo "JOB Name:$SLURM_JOB_NAME"
    echo "Array job ID: $SLURM_ARRAY_JOB_ID"
    echo "====================================="
    echo " "
    echo "Running with variables:"
    echo "  SETTINGS=$SETTINGS"
    echo "  FILTERS=$FILTERS"
    echo "  ADVANCED=$ADVANCED"
    echo " "
}
fi
echo "Start - Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running the BindCraft pipeline for TASK_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"


#################
# RUN BINDCRAFT #
#################

python -u "/home/kunzj/${BindCraft_dir}/bindcraft.py" --settings "${SETTINGS}" --filters "${FILTERS}" --advanced "${ADVANCED}" > "$PYTHON_LOG" 2>&1



