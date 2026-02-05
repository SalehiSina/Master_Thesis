#!/bin/bash
# ^Batch script starts with shebang line

#SBATCH --ntasks=1                              #
#SBATCH --nodes=1                               #
#SBATCH --cpus-per-task=4                       #
#SBATCH --gres=gpu:2                            #
#SBATCH --time=02:00:00                         #
#SBATCH --account=p_scads_celltracking          #
#SBATCH --job-name=Steamboat_training           # All #SBATCH lines have to follow uninterrupted
#SBATCH --output=steamboat_train-%j.out    # after the shebang line
#SBATCH --error=steamboat_train-%j.err     # Comments start with # and do not count as interruptions


python Ex3_crc_train.py
