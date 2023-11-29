#!/bin/bash

GPUS=1
MASTER_PORT=41043
OUTPUT_DIR='OUTPUT/pretraining'
DATA_PATH='official_hmdb_splits1'
DATA_ROOT='hmdb51_mp4'

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --master_port ${MASTER_PORT} --nnodes=1 \
    --node_rank=0 --master_addr=localhost \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set SSV2 --nb_classes 174 \
    --data_path ${DATA_PATH} \
    --data_root ${DATA_ROOT} \
    --finetune video_teacher.pth \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 --short_side_size 224 \
    --opt adamw --opt_betas 0.9 0.999 --weight_decay 0.05 \
    --batch_size 24 --update_freq 1 --num_sample 2 \
    --save_ckpt_freq 5 --no_save_best_ckpt \
    --num_frames 16 \
    --lr 5e-4 --epochs 30 \
    --dist_eval --test_num_segment 2 --test_num_crop 3 \
    --use_checkpoint \
    --enable_deepspeed --data_set HMDB51