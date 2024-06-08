#!/bin/bash
#SBATCH -c 12
#SBATCH -N 1
#SBATCH --mem-per-cpu=30G
#SBATCH -t 3-00:00:00
#SBATCH -o gen_unsafe.out
#SBATCH -A dplab
#SBATCH -p gpu
#SBATCH --gres gpu:a100:1 

module load ffmpeg
module load anaconda
source activate
conda activate /scratch/trv3px/video_detection/AnimateDiff/env

python -m scripts.animate --config configs/prompts/v3/v3-2-animation-RealisticVision.yaml