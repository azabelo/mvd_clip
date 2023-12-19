import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict
from timm.models import create_model
from optim_factory import create_optimizer
from datasets import build_distillation_dataset
from engine_for_pretraining import train_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from utils import multiple_pretrain_samples_collate
import modeling_student
import modeling_teacher
import modeling_video_teacher

# MY CHANGES
import wandb
import clip
from rei.eva_clip import create_model_and_transforms, get_tokenizer
import modeling_finetune_v2 #videomaev2 teacher
import torch.nn as nn
import copy
# END MY CHANGES


def get_args():
    parser = argparse.ArgumentParser('MVD pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--update_freq', default=1, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_masked_video_student_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--decoder_depth', default=2, type=int,
                        help='depth of decoder')

    parser.add_argument('--image_teacher_model', default='mae_teacher_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of teacher model')
    parser.add_argument('--image_teacher_model_ckpt_path', default='', type=str)
    parser.add_argument('--distillation_target_dim', default=768, type=int)
    parser.add_argument('--distill_loss_func', default='SmoothL1', choices=['L1', 'L2', 'SmoothL1'],
                        type=str)
    parser.add_argument('--image_teacher_loss_weight', default=1.0, type=float)

    parser.add_argument('--video_teacher_model', default='pretrain_videomae_teacher_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of teacher model')
    parser.add_argument('--video_teacher_model_ckpt_path', default='', type=str)
    parser.add_argument('--video_distillation_target_dim', default=768, type=int)
    parser.add_argument('--video_distill_loss_func', default='SmoothL1', choices=['L1', 'L2', 'SmoothL1'],
                        type=str)
    parser.add_argument('--video_teacher_loss_weight', default=1.0, type=float)
    parser.add_argument('--video_teacher_drop_path', default=0., type=float)

    parser.add_argument('--teacher_input_size', default=224, type=int,
                        help='videos input size for backbone')
    parser.add_argument('--video_teacher_input_size', default=224, type=int,
                        help='videos input size for backbone')

    parser.add_argument('--feat_decoder_embed_dim', default=None, type=int)
    parser.add_argument('--feat_decoder_num_heads', default=None, type=int)

    parser.add_argument('--norm_feature', action='store_true', default=False)

    parser.add_argument('--tubelet_size', default=2, type=int)

    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--use_checkpoint', action='store_true', default=False)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.6, metavar='PCT',
                        help='Color jitter factor (default: 0.6)')
    parser.add_argument('--color_jitter_hue', type=float, default=0.15, metavar='PCT',
                        help='Color jitter Hue factor (default: 0.15)')
    parser.add_argument('--gray_scale', type=float, default=0.2, metavar='PCT',
                        help='Gray scale factor (default: 0.2)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--num_sample', type=int, default=1,
                        help='Repeated_aug (default: 1)')

    # Dataset parameters
    parser.add_argument('--data_root', default=None, type=str,
                        help='dataset path root')
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='path of dataset file list')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--load_model', default=None, help='init from checkpoint')

    parser.add_argument('--use_cls_token', action='store_true', default=False)
    parser.add_argument('--time_stride_loss', action='store_true', default=True,
                        help='predict one frame per temporal stride if true')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


    # MY CHANGES
    parser.add_argument('--knn_freq', default=10, type=int)
    parser.add_argument('--use_wandb', default=0, type=int)
    parser.add_argument('--wandb_project_name', default='no_name', type=str)
    parser.add_argument('--notes_for_wandb_run', default='', type=str)
    parser.add_argument('--cls', default=0, type=int)
    parser.add_argument('--resume_checkpoint', default='none')

    args_ret = parser.parse_args()
    if args_ret.cls == 1:
        args_ret.use_cls_token = True
    if args_ret.resume_checkpoint != 'none':
        args_ret.resume = args_ret.resume_checkpoint

    return args_ret
    # END MY_CHANGES


def get_image_teacher_model(args):
    print("using DEFAULT image teacher")
    model = create_model(
        args.image_teacher_model,
        pretrained=False,
        img_size=args.teacher_input_size,
    )
    return model

# MY CHANGES

def get_clip_model(args):
    print("using CLIP image teacher")
    args.image_teacher_model = 'vit_base_patch16_224'
    model, preprocess = clip.load("ViT-B/16", device=args.device)
    # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # note: you haven't checked clip for correctness yet (you made changes)
    return model.visual

def get_eva_clip_model(args):
    print("using EVA CLIP image teacher")
    args.image_teacher_model = 'vit_base_patch16_224'
    pretrained = 'eva_clip_model.pth'
    model_name = "EVA02-CLIP-B-16"
    model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
    # does this have its own normalization?
    # note: you haven't checked rei for correctness yet (you made changes)
    return model.visual

# END MY CHANGES

def get_video_teacher_model(args):
    print(f"Creating teacher model: {args.video_teacher_model}")
    model = create_model(
        args.video_teacher_model,
        pretrained=False,
        img_size=args.video_teacher_input_size,
        drop_path_rate=args.video_teacher_drop_path,
    )
    return model

# MY CHANGES

def get_videomaev2_model(args):
    # for now this is the same as video teacher, but perhaps doing it more directly like you did
    # with CLIP would solve your problems - no not rly, its complicated

    print(f"Creating teacher model: {args.video_teacher_model}")
    model = create_model(
        args.video_teacher_model,
        pretrained=False,
        img_size=args.video_teacher_input_size,
        drop_path_rate=args.video_teacher_drop_path,
    )
    return model

    #this works but is pretraining
    # model = create_model(
    #     'pretrain_videomae_base_patch16_224',
    #     pretrained=False,
    #
    #     drop_path_rate=args.video_teacher_drop_path,
    # # )
    # model = create_model(
    #     'vit_base_patch16_224',
    #     img_size=224,
    #     pretrained=False,
    #     num_classes=10,
    #     all_frames=args.num_frames,
    #     tubelet_size=args.tubelet_size,
    #     drop_rate=0,
    #     drop_path_rate=0,
    #     attn_drop_rate=0,
    #     head_drop_rate=0,
    #     drop_block_rate=None,
    # )
    # #took args to be the same as videomaev2 repo
    # # model = create_model(
    # #     'pretrain_videomae_base_patch16_224',
    # #     pretrained=False,
    # #     drop_path_rate=0,
    # #     drop_block_rate=None,
    # #     all_frames=16,
    # #     tubelet_size=args.tubelet_size,
    # #     decoder_depth=0,
    # #     with_cp=args.with_checkpoint)
    # return model

def get_checkpoint_model(args):
    # for now this is the same as video teacher, but perhaps doing it more directly like you did
    # with CLIP would solve your problems

    print(f"Creating teacher model: {args.video_teacher_model}")
    model = create_model(
        args.video_teacher_model,
        pretrained=False,
        img_size=args.video_teacher_input_size,
        drop_path_rate=args.video_teacher_drop_path,
    )
    return model

def get_video_clip_teacher_model(args):
    pass

# END MY CHANGES

def get_model(args):
    print(f"Creating model: {args.model}")

    # could experiment with giving checkpoint path here, but I think a dif part of the code might handle that
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth,
        use_cls_token=args.use_cls_token,
        num_frames=args.num_frames,
        target_feature_dim=args.distillation_target_dim,
        target_video_feature_dim=args.video_distillation_target_dim,
        feat_decoder_embed_dim=args.feat_decoder_embed_dim,
        feat_decoder_num_heads=args.feat_decoder_num_heads,
        use_checkpoint=args.use_checkpoint,
        tubelet_size=args.tubelet_size,
    )

    return model


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    dataset_train = build_distillation_dataset(args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_pretrain_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker,
        collate_fn=collate_func,
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ## MY CHANGES

    image_teacher_model = None
    ## DEFAULT IMAGE TEACHER ##
    if args.image_teacher_model_ckpt_path == 'image_teacher.pth':
        image_teacher_model = get_image_teacher_model(args)

        if args.image_teacher_model_ckpt_path:
            if args.image_teacher_model_ckpt_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.image_teacher_model_ckpt_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.image_teacher_model_ckpt_path, map_location='cpu')

            print("Load teacher ckpt from %s" % args.image_teacher_model_ckpt_path)
            checkpoint_model = None
            for model_key in args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load state_dict by model_key = %s" % model_key)
                    break

            if checkpoint_model is None:
                checkpoint_model = checkpoint

            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]
                elif 'pos_embed' in key:
                    continue
                else:
                    new_dict[key] = checkpoint_model[key]
            checkpoint_model = new_dict

            utils.load_state_dict(image_teacher_model, checkpoint_model, prefix=args.model_prefix)
    ## CLIP ##
    elif args.image_teacher_model_ckpt_path == 'clip_model.pth':
        image_teacher_model = get_clip_model(args)
    ## EVA-CLIP ## untested
    elif args.image_teacher_model_ckpt_path == 'eva_clip_model.pth':
        image_teacher_model = get_eva_clip_model(args)
    ## INVALID ##
    else:
        print("Invalid image teacher model ckpt path")
        exit(1)

    image_teacher_model.to(device)

    ## END MY CHANGES ##

    ## MY CHANGES ##

    video_teacher_model = None
    ## DEFAULT VIDEO TEACHER ##
    if args.video_teacher_model_ckpt_path == 'video_teacher.pth':
        video_teacher_model = get_video_teacher_model(args)

        if args.video_teacher_model_ckpt_path:
            if args.video_teacher_model_ckpt_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.video_teacher_model_ckpt_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.video_teacher_model_ckpt_path, map_location='cpu')

            print("Load video teacher ckpt from %s" % args.video_teacher_model_ckpt_path)
            checkpoint_model = None
            for model_key in args.model_key.split('|'):
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print("Load video state_dict by model_key = %s" % model_key)
                    break

            if checkpoint_model is None:
                checkpoint_model = checkpoint

            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('backbone.'):
                    new_dict[key[9:]] = checkpoint_model[key]
                elif 'pos_embed' in key:
                    continue
                else:
                    new_dict[key] = checkpoint_model[key]
            checkpoint_model = new_dict

            utils.load_state_dict(video_teacher_model, checkpoint_model, prefix=args.model_prefix)
    ## VideoMAEv2 old
    # elif args.video_teacher_model_ckpt_path == 'videoMAEv2_model.pth':
    #     video_teacher_model = get_videomaev2_model(args)
    #
    #     checkpoint = torch.load(args.video_teacher_model_ckpt_path, map_location='cpu')
    #     checkpoint = checkpoint['module']
    #
    #     print("Load video teacher ckpt from %s" % args.video_teacher_model_ckpt_path)
    #     checkpoint_model = None
    #     for model_key in args.model_key.split('|'):
    #         if model_key in checkpoint:
    #             checkpoint_model = checkpoint[model_key]
    #             print("Load video state_dict by model_key = %s" % model_key)
    #             break
    #
    #     if checkpoint_model is None:
    #         checkpoint_model = checkpoint
    #
    #     for k in ['head.weight', 'head.bias']:
    #         if k in checkpoint_model:
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]
    #
    #     all_keys = list(checkpoint_model.keys())
    #     new_dict = OrderedDict()
    #
    #     for key in all_keys:
    #         print("v2 teacher: ", key)
    #         if key == 'fc_norm.weight':
    #             new_dict["encoder.norm.weight"] = checkpoint_model[key]
    #             continue
    #         elif key == 'fc_norm.bias':
    #             new_dict["encoder.norm.bias"] = checkpoint_model[key]
    #             continue
    #
    #         if key.startswith('backbone.'):
    #             new_dict[key[9:]] = checkpoint_model[key]
    #         elif 'pos_embed' in key:
    #             continue
    #         else:
    #             new_dict["encoder." + key] = checkpoint_model[key]
    #
    #     checkpoint_model = new_dict
    #     utils.load_state_dict(video_teacher_model, checkpoint_model, prefix=args.model_prefix)
    #     for param in video_teacher_model.parameters():
    #         param.requires_grad_(False)
    ## VideoMAEv2 new
    elif args.video_teacher_model_ckpt_path == 'videoMAEv2_model.pth':
        video_teacher_model = get_videomaev2_model(args)

        checkpoint = torch.load(args.video_teacher_model_ckpt_path, map_location='cpu')
        checkpoint = checkpoint['module']

        print("Load video teacher ckpt from %s" % args.video_teacher_model_ckpt_path)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load video state_dict by model_key = %s" % model_key)
                break

        if checkpoint_model is None:
            checkpoint_model = checkpoint

        # not sure why the default doesnt have it and therefore doesn't need to remove,
        # but i think its better to keep it for v2
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()

        for key in all_keys:
            print("v2 teacher: ", key)
            if key == 'fc_norm.weight':
                new_dict["encoder.norm.weight"] = checkpoint_model[key]
                continue
            elif key == 'fc_norm.bias':
                new_dict["encoder.norm.bias"] = checkpoint_model[key]
                continue
            elif key == 'head.weight' or key == 'head.bias':
                new_dict[key] = checkpoint_model[key]
                continue
            elif 'pos_embed' in key:
                continue
            else:
                new_dict["encoder."+key] = checkpoint_model[key]

        checkpoint_model = new_dict
        utils.load_state_dict(video_teacher_model, checkpoint_model, prefix=args.model_prefix)
        for param in video_teacher_model.parameters():
            param.requires_grad_(False)
    ## Pretrained Checkpoint (using mvd)
    elif 'checkpoint' in args.video_teacher_model_ckpt_path:
        # this way is bad because the entire model seems messed up when no cls token
        # video_teacher_model = get_checkpoint_model(args)
        #
        # checkpoint = torch.load(args.video_teacher_model_ckpt_path, map_location='cpu')
        #
        # print("Load video teacher ckpt from %s" % args.video_teacher_model_ckpt_path)
        # checkpoint_model = None
        # for model_key in args.model_key.split('|'):
        #     if model_key in checkpoint:
        #         checkpoint_model = checkpoint[model_key]
        #         print("Load video state_dict by model_key = %s" % model_key)
        #         break
        #
        # if checkpoint_model is None:
        #     checkpoint_model = checkpoint
        #
        # for k in ['head.weight', 'head.bias']:
        #     if k in checkpoint_model:
        #         print(f"Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]
        #
        # all_keys = list(checkpoint_model.keys())
        # new_dict = OrderedDict()
        #
        # for key in all_keys:
        #     if key.startswith('backbone.'):
        #         new_dict[key[9:]] = checkpoint_model[key]
        #     elif 'pos_embed' in key:
        #         continue
        #     else:
        #         new_dict["encoder."+key] = checkpoint_model[key]
        #
        # checkpoint_model = new_dict
        # utils.load_state_dict(video_teacher_model, checkpoint_model, prefix=args.model_prefix)
        #
        # video_teacher_model = video_teacher_model

        temp_model = create_model(
            args.model,
            pretrained=False,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            decoder_depth=args.decoder_depth,
            use_cls_token=True,# true for 4799 models
            num_frames=args.num_frames,
            target_feature_dim=args.distillation_target_dim,
            target_video_feature_dim=args.video_distillation_target_dim,
            feat_decoder_embed_dim=args.feat_decoder_embed_dim,
            feat_decoder_num_heads=args.feat_decoder_num_heads,
            use_checkpoint=args.use_checkpoint,
            tubelet_size=args.tubelet_size,
        )
        args_copy = copy.deepcopy(args)
        args_copy.resume_checkpoint = args.video_teacher_model_ckpt_path
        optimizer_temp = create_optimizer(
            args_copy, model_without_ddp)
        loss_scaler_temp = NativeScaler()
        utils.auto_load_model(
            args=args_copy, model=temp_model, model_without_ddp=None, optimizer=optimizer_temp,
            loss_scaler=loss_scaler_temp, model_ema=None
        )
        temp_model.to(device)
        temp_model.eval()
        class Teacher_from_Student(nn.Module):
            def __init__(self):
                super(Teacher_from_Student, self).__init__()

            def forward(self, x):
                # Calls forward encoder of the student model
                temp_model.eval()
                with torch.no_grad():
                    temp_model.eval()
                    empty_mask = torch.zeros((x.shape[0], 1568), dtype=torch.bool).to(x.device)
                    encoded_output = temp_model.forward_encoder(x, empty_mask)
                    encoded_output = encoded_output[:, 1:, :] # remove cls token
                return encoded_output

        video_teacher_model = Teacher_from_Student()
        del loss_scaler_temp
        del optimizer_temp
        del args_copy

    ## INVALID
    else:
        print("Invalid video teacher model ckpt path")
        exit(1)
    video_teacher_model.to(device)


    ## END MY CHANGES ##

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * args.num_sample * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // int(total_batch_size / args.num_sample)
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "config.txt"), mode="a", encoding="utf-8") as f:
            for arg in vars(args):
                f.write(format(arg, '<20') + " " + format(str(getattr(args, arg)), '<') + "\n")

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        loss_scaler=loss_scaler, model_ema=None
    )

    # MY CHANGES
    if args.use_wandb != 0:
        run_name = f"pretrain {args.notes_for_wandb_run}  image_teacher: {args.image_teacher_model_ckpt_path}, video_teacher:{args.video_teacher_model_ckpt_path} bs: {args.batch_size}, update: {args.update_freq}, lr: {args.lr}, epochs: {args.epochs}, \
 warmup: {args.warmup_epochs}, "
        print(run_name)
        wandb.init(project=args.wandb_project_name, name=run_name)
        wandb.config.update(args)
    # END MY CHANGES

    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)

        train_stats = train_one_epoch(
            args, model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            update_freq=args.update_freq,
            time_stride_loss=True,
            image_teacher_model=image_teacher_model,
            video_teacher_model=video_teacher_model,
            norm_feature=args.norm_feature,
        )

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=None)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
