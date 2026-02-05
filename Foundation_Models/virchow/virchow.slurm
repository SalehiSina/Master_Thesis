#!/bin/bash
# ^Batch script starts with shebang line

#SBATCH --ntasks=1                              #
#SBATCH --nodes=1                               #
#SBATCH --cpus-per-task=3                       #
#SBATCH --gres=gpu:3                            #
#SBATCH --time=1:30:00                         #
#SBATCH --account=p_scads_celltracking          #
#SBATCH --job-name=Embedding                 # All #SBATCH lines have to follow uninterrupted
#SBATCH --output=virchow-%j.out               # after the shebang line
#SBATCH --error=virchow-%j.err                # 


torchrun --nproc_per_node=3 ./virchow/Embedding.py