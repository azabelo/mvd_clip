#!/bin/bash
GPUS=`nvidia-smi -L | wc -l`
OUTPUT_DIR='OUTPUT/mvd_vit_large_with_vit_large_teacher_k400_epoch_400'
DATA_PATH='k400_anno/train.csv'
DATA_ROOT='your_path/kinetics400'

# train on 32 V100 GPUs (4 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=${GPUS} \
        --master_port ${MASTER_PORT} --nnodes=${NODE_COUNT} \
        --node_rank=${RANK} --master_addr=${MASTER_ADDR} \
        run_mvd_pretraining.py \
        --data_path ${DATA_PATH} \
        --data_root ${DATA_ROOT} \
        --model pretrain_masked_video_student_large_patch16_224 \
        --opt adamw --opt_betas 0.9 0.95 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --image_teacher_model mae_teacher_vit_large_patch16 \
        --distillation_target_dim 1024 \
        --distill_loss_func SmoothL1 \
        --image_teacher_model_ckpt_path 'your_path/mae_pretrain_vit_large.pth' \
        --video_teacher_model pretrain_videomae_teacher_large_patch16_224 \
        --video_distillation_target_dim 1024 \
        --video_distill_loss_func SmoothL1 \
        --video_teacher_model_ckpt_path 'your_path/k400_videomae_pretrain_large_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600.pth' \
        --mask_type tube --mask_ratio 0.9 \
        --decoder_depth 4 --feat_decoder_embed_dim 512 --feat_decoder_num_heads 8 \
        --batch_size 8 --update_freq 2 --save_ckpt_freq 10 \
        --num_frames 16 --sampling_rate 4 \
        --lr 1.5e-4 --min_lr 1e-4 --drop_path 0.1 --warmup_epochs 40 --epochs 401 \
        --auto_resume
