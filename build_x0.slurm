#!/bin/bash
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --mem-per-cpu=40G
#SBATCH -t 3-00:00:00
#SBATCH -o build_x0_1.out
#SBATCH -A dplab
#SBATCH -p gpu
#SBATCH --gres gpu:a100:1 

module load ffmpeg
module load anaconda
source activate
conda activate /scratch/trv3px/video_detection/AnimateDiff/env

python build_x0.py --config configs/prompts/v3/v3-1-T2V.yaml