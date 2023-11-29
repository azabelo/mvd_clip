#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 17 ]; then
  echo "Usage: $0 GPUS PORT IMAGE_TEACHER_PATH VIDEO_TEACHER_PATH DATA_DIRECTORY TRAIN_CSV EPOCHS BATCH_SIZE LEARNING_RATE WARMUP UPDATE_FREQ SAVE_FREQ KNN_FREQ USE_WANDB WANDB_PROJECT_NAME NOTES_FOR_WANDB_RUN USE_CLS_TOKEN"
  exit 1
fi

# Assign command line arguments to variables
GPUS=$1
PORT=$2
IMAGE_TEACHER_PATH=$3
VIDEO_TEACHER_PATH=$4
DATA_DIRECTORY=$5
TRAIN_CSV=$6
EPOCHS=$7
BATCH_SIZE=$8
LEARNING_RATE=$9
WARMUP=${10}
UPDATE_FREQ=${11}
SAVE_FREQ=${12}
KNN_FREQ=${13}
USE_WANDB=${14}
WANDB_PROJECT_NAME=${15}
NOTES_FOR_WANDB_RUN=${16}
USE_CLS_TOKEN=${17}
OUTPUT_DIR='OUTPUT/pretrain/'

# dont forget that k400 and smaller dataset like UCF101 use different parameters

OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=${GPUS} \
        --master_port ${PORT} --nnodes=1 \
        --node_rank=0 --master_addr=localhost \
        run_mvd_pretraining.py \
        --data_path ${TRAIN_CSV} \
        --data_root ${DATA_DIRECTORY} \
        --model pretrain_masked_video_student_base_patch16_224 \
        --opt adamw --opt_betas 0.9 0.95 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --image_teacher_model mae_teacher_vit_base_patch16 \
        --distillation_target_dim 768 \
        --distill_loss_func SmoothL1 \
        --image_teacher_model_ckpt_path ${IMAGE_TEACHER_PATH} \
        --video_teacher_model pretrain_videomae_teacher_base_patch16_224 \
        --video_distillation_target_dim 768 \
        --video_distill_loss_func SmoothL1 \
        --video_teacher_model_ckpt_path ${VIDEO_TEACHER_PATH} \
        --mask_type tube --mask_ratio 0.9 --decoder_depth 2 \
        --batch_size ${BATCH_SIZE} --update_freq ${UPDATE_FREQ} --save_ckpt_freq ${SAVE_FREQ} \
        --num_frames 16 --sampling_rate 4 \
        --lr ${LEARNING_RATE} --min_lr 1e-4 --drop_path 0.1 --warmup_epochs ${WARMUP} --epochs ${EPOCHS} \
        --auto_resume --norm_feature \
        --knn_freq ${KNN_FREQ} --use_wandb ${USE_WANDB} --wandb_project_name ${WANDB_PROJECT_NAME} \
        --notes_for_wandb_run ${NOTES_FOR_WANDB_RUN} --cls ${USE_CLS_TOKEN}