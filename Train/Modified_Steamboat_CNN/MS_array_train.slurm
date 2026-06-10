#!/bin/bash
# ^Batch script starts with shebang line

#SBATCH --nodes=1                               #
#SBATCH --cpus-per-task=1                       #
#SBATCH --gres=gpu:1                            #
#SBATCH --time=5:30:00                          #
#SBATCH --account=p_scads_celltracking          #
#SBATCH --job-name=train_steamboat_CNN          #
#SBATCH --mem=54G

#SBATCH --output=HPC_logs/out_%A.txt
#SBATCH --error=HPC_logs/err_%A.txt


# -----------------------------------
# parameters
# -----------------------------------

MASK_RATE=0.0
SEED=0

# -----------------------------------
# Launch training
# -----------------------------------
python Train/Modified_Steamboat_CNN/Xenium_Breast_Cancer_train.py \
    --mask_rate $MASK_RATE \
    --seed $SEED