#!/bin/bash

# they don't use warmup during finetuning

# Check if the correct number of arguments is provided
if [ "$#" -ne 16 ]; then
  echo "Usage: $0 GPUS PORT CHECKPOINT_PATH DATA_DIRECTORY CSV_DIR EPOCHS BATCH_SIZE LEARNING_RATE WARMUP UPDATE_FREQ SAVE_FREQ KNN_FREQ USE_WANDB WANDB_PROJECT_NAME NOTES_FOR_WANDB_RUN USE_CLS_TOKEN"
  echo " "
  echo "Example: $0 1 12345 video_teacher.pth hmdb51_mp4 official_hmdb_splits1 101 16 5e-4 5 4 10 10 1 project_name wandb_notes 1"
  exit 1
fi


# Assign command line arguments to variables
GPUS=$1
PORT=$2
CHECKPOINT_PATH=$3
DATA_DIRECTORY=$4
CSV_DIR=$5
EPOCHS=$6
BATCH_SIZE=$7
LEARNING_RATE=$8
WARMUP=${9}
UPDATE_FREQ=${10}
SAVE_FREQ=${11}
KNN_FREQ=${12}
USE_WANDB=${13}
WANDB_PROJECT_NAME=${14}
NOTES_FOR_WANDB_RUN=${15}
USE_CLS_TOKEN=${16}
OUTPUT_DIR='OUTPUT/finetune/'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --master_port ${PORT} --nnodes=1 \
    --node_rank=0 --master_addr=localhost \
    run_alignment.py  \
    --model vit_base_patch16_224 \
    --data_set HMDB51 --nb_classes 51 \
    --data_path ${CSV_DIR} \
    --data_root ${DATA_DIRECTORY} \
    --finetune ${CHECKPOINT_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --batch_size ${BATCH_SIZE} --update_freq ${UPDATE_FREQ} --num_sample 1 \
    --save_ckpt_freq ${SAVE_FREQ} --no_save_best_ckpt \
    --num_frames 16 \
    --lr ${LEARNING_RATE} --epochs ${EPOCHS} \
    --dist_eval --test_num_segment 1 --test_num_crop 1 \
    --use_checkpoint \
    --enable_deepspeed \
    --warmup_epochs ${WARMUP} --knn_freq ${KNN_FREQ} --use_wandb ${USE_WANDB} \
    --wandb_project_name ${WANDB_PROJECT_NAME} \
    --notes_for_wandb_run ${NOTES_FOR_WANDB_RUN} --cls ${USE_CLS_TOKEN}
