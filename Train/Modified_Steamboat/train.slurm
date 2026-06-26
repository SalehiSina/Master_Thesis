#!/bin/bash
# ^Batch script starts with shebang line

#SBATCH --nodes=1                               #
#SBATCH --cpus-per-task=1                       #
#SBATCH --gres=gpu:1                            #
#SBATCH --time=3:30:00                          #
#SBATCH --account=mobil-08                      #
#SBATCH --job-name=train_MS                     #
#SBATCH --mem=8G

#SBATCH --output=HPC_logs/out_%A.txt
#SBATCH --error=HPC_logs/err_%A.txt


# -----------------------------------
# parameters
# -----------------------------------

MASK_RATE=0.8
SEED=0

# -----------------------------------
# Launch training
# -----------------------------------
python Train/Modified_Steamboat/Xenium_Breast_Cancer_train.py \
    --mask_rate $MASK_RATE \
    --seed $SEED