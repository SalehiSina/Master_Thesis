#!/bin/bash
# ^Batch script starts with shebang line

#SBATCH --nodes=1                               #
#SBATCH --cpus-per-task=4                       #
#SBATCH --gres=gpu:1                            #
#SBATCH --time=12:00:00                          #
#SBATCH --account=mobil-08                      #
#SBATCH --job-name=AE                           #
#SBATCH --mem=32G

#SBATCH --output=HPC_logs/out_%A.txt
#SBATCH --error=HPC_logs/err_%A.txt


# -----------------------------------
# parameters
# -----------------------------------

N_Epochs=20
SEED=0

# -----------------------------------
# Launch training
# -----------------------------------
python ./HistoAEs/Breast\ Cancer/AE_training.py \
    --n_epochs $N_Epochs \
    --seed $SEED