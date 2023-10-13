#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --cpus-per-gpu=2        # Number of CPU cores to allocate
#SBATCH --export=ALL
#SBATCH --gres=gpu:4             # Number of GPU's
#SBATCH --mem-per-gpu=20G	 # Memory to allocate in MB per allocated CPU core
#SBATCH --partition=gpu

echo "Start training"

conda activate astod

PORT=$(( $RANDOM % 50 + 10000 ))

torchrun --nproc_per_node=4 --master_addr 127.0.0.5 --master_port $PORT src/train.py \
         --config configs/faster_rcnn_coco_finetune.yaml \
         --train-file coco/annotations/instances_train2017.$1@$2.json \
         --output-dir models/faster_rcnn/coco/$2/finetune_1/model_$1_21_glob \
         --resume models/faster_rcnn/coco/$2/student_1/model_$1_21_glob/best_model.pth \
         --split $2 \
         --seed $1