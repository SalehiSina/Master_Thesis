#!/bin/bash
# ^Batch script starts with shebang line

#SBATCH --ntasks=1                              #
#SBATCH --nodes=1                               #
#SBATCH --cpus-per-task=3                       #
#SBATCH --gres=gpu:3                            #
#SBATCH --time=2:30:00                         #
#SBATCH --account=p_scads_celltracking          #
#SBATCH --job-name=Embedding                 # All #SBATCH lines have to follow uninterrupted
#SBATCH --output=h_optimus_0-%j.out               # after the shebang line
#SBATCH --error=h_optimus_0-%j.err                # 


torchrun --nproc_per_node=3 ./h_optimus/Embedding.py