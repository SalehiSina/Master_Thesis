#!/bin/bash
# ^Batch script starts with shebang line

#SBATCH --nodes=1                               #
#SBATCH --cpus-per-task=1                       #
#SBATCH --gres=gpu:1                            #
#SBATCH --time=1:30:00                          #
#SBATCH --account=p_scads_celltracking          #
#SBATCH --job-name=train_NOISE                  #
#SBATCH --mem=8G

#SBATCH --output=HPC_logs/out_%A_%a.txt
#SBATCH --error=HPC_logs/err_%A_%a.txt

# 55 total jobs, max 4 simultaneously
#SBATCH --array=17-54%4


# -----------------------------------
# parameters
# -----------------------------------
MASKING_RATES=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0)
NUM_REPEATS=5

TASK_ID=$SLURM_ARRAY_TASK_ID
MASK_INDEX=$((TASK_ID / NUM_REPEATS))
SEED=$((TASK_ID % NUM_REPEATS))

MASK_RATE=${MASKING_RATES[$MASK_INDEX]}



echo "TASK_ID      = $TASK_ID"
echo "MASK_RATE    = $MASK_RATE"
echo "SEED         = $SEED"

# -----------------------------------
# Launch training
# -----------------------------------
python Notebooks/GNN_Cross/Xenium_Breast_Cancer_train_NOISE.py \
    --mask_rate $MASK_RATE \
    --seed $SEED