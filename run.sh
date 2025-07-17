#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=eval_qnorm_0.9_temporal_overlay
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output=train_output_%x_%A.out

module purge
module load 2024

cd "/home/jlin1/OutlierDetection/" || exit

# source .venv/bin/activate

# python VAD/src/vad/data_shanghai.py
# python unified_pipeline_runner.py
python /home/jlin1/OutlierDetection/eval/single_file_eval.py
# python /home/jlin1/OutlierDetection/visual.py