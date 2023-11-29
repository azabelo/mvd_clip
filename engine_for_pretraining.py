import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import torchvision
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# MY CHANGES
import wandb
from datasets import build_dataset
from utils import multiple_samples_collate
from functools import partial
import copy
import torch.optim as optim
# END MY CHANGES

Loss_func_choice = {'L1': torch.nn.L1Loss, 'L2': torch.nn.MSELoss, 'SmoothL1': torch.nn.SmoothL1Loss}


def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, update_freq=None, time_stride_loss=True, lr_scale=1.0,
                    image_teacher_model=None, video_teacher_model=None, norm_feature=False):

    # MY CHANGES
    # not sure if the pretraining accuracy stuff needs normalization
    if epoch % args.knn_freq == 0:
        pretraining_accuracy(model, args)
    # END MY CHANGES

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    LN_img = nn.LayerNorm(args.distillation_target_dim, eps=1e-6, elementwise_affine=False).cuda()
    LN_vid = nn.LayerNorm(args.video_distillation_target_dim, eps=1e-6, elementwise_affine=False).cuda()

    loss_func_img_feat = Loss_func_choice[args.distill_loss_func]()
    loss_func_vid_feat = Loss_func_choice[args.video_distill_loss_func]()
    image_loss_weight = args.image_teacher_loss_weight
    video_loss_weight = args.video_teacher_loss_weight

    tubelet_size = args.tubelet_size

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        update_step = step // update_freq
        it = start_steps + update_step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None and step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"] * lr_scale
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, videos_for_teacher, bool_masked_pos = batch
        videos = videos.to(device, non_blocking=True)
        videos_for_teacher = videos_for_teacher.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        _, _, T, _, _ = videos.shape

        with torch.cuda.amp.autocast():
            output_features, output_video_features = model(videos, bool_masked_pos)
            with torch.no_grad():
                image_teacher_model.eval()
                if time_stride_loss:
                    teacher_features = image_teacher_model(
                        rearrange(videos_for_teacher[:, :, ::tubelet_size, :, :], 'b c t h w -> (b t) c h w'),
                    )
                    teacher_features = rearrange(teacher_features, '(b t) l c -> b (t l) c', t=T//tubelet_size)
                else:
                    teacher_features = image_teacher_model(
                        rearrange(videos_for_teacher, 'b c t h w -> (b t) c h w'),
                    )
                    teacher_features = rearrange(teacher_features, '(b t d) l c -> b (t l) (d c)', t=T//tubelet_size, d=tubelet_size)
                if norm_feature:
                    teacher_features = LN_img(teacher_features)

                video_teacher_model.eval()
                videos_for_video_teacher = videos if args.video_teacher_input_size == args.input_size \
                    else videos_for_teacher

                video_teacher_features = video_teacher_model(videos_for_video_teacher)
                if norm_feature:
                    video_teacher_features = LN_vid(video_teacher_features)

            B, _, D = output_features.shape
            loss_img_feat = loss_func_img_feat(
                input=output_features,
                target=teacher_features[bool_masked_pos].reshape(B, -1, D)
            )
            loss_value_img_feat = loss_img_feat.item()

            B, _, D = output_video_features.shape
            loss_vid_feat = loss_func_vid_feat(
                input=output_video_features,
                target=video_teacher_features[bool_masked_pos].reshape(B, -1, D)
            )
            loss_value_vid_feat = loss_vid_feat.item()

            loss = image_loss_weight * loss_img_feat + video_loss_weight * loss_vid_feat

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(step + 1) % update_freq == 0)
        if (step + 1) % update_freq == 0:
            optimizer.zero_grad()
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_img_feat=loss_value_img_feat)
        metric_logger.update(loss_vid_feat=loss_value_vid_feat)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_img_feat=loss_value_img_feat, head="loss_img_feat")
            log_writer.update(loss_vid_feat=loss_value_vid_feat, head="loss_vid_feat")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

            # MY CHANGES
            wandb.log(
                {"epoch": epoch, "batch": step, "train_loss": loss_value, " train_img_feat_loss": loss_value_img_feat,
                 "min_lr": min_lr, "max_lr": max_lr, "train_vid_feat_loss": loss_value_vid_feat,
                 "grad_norm": grad_norm, "loss_scale": loss_scale_value, "weight_decay": weight_decay_value,
                 "lr_scale": lr_scale, })
            # END MY CHANGES

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def pretraining_accuracy(model, args):
    model.eval()

    # args that are only present in finetuning were copied over
    args_copy = copy.deepcopy(args)
    args_copy.data_set = 'HMDB51'
    args_copy.nb_classes = 51
    args_copy.data_path = 'official_hmdb_splits1'
    args_copy.num_frames = 16
    args_copy.short_side_size = 224
    args_copy.aa = 'rand-m7-n4-mstd0.5-inc1'
    args_copy.remode = 'pixel'
    args_copy.recount = 1
    args_copy.reprob = 0.25
    args_copy.sampling_rate = 4
    args_copy.test_num_segment = 2
    args_copy.test_num_crop = 3

    dataset_train, args_copy.nb_classes = build_dataset(is_train=True, test_mode=False, args=args_copy)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    if args_copy.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args_copy.batch_size,
        num_workers=args_copy.num_workers,
        pin_memory=args_copy.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
    )

    class LinearClassifier(nn.Module):
        def __init__(self):
            super(LinearClassifier, self).__init__()
            self.fc = nn.Linear(1569 * 768, 51)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten the input
            x = self.fc(x)
            return x

    # Instantiate the model
    linear_model = LinearClassifier()
    linear_criterion = nn.CrossEntropyLoss()
    linear_optimizer = optim.SGD(linear_model.parameters(), lr=0.01)

    class TwoLayerClassifier(nn.Module):
        def __init__(self):
            super(TwoLayerClassifier, self).__init__()
            self.fc1 = nn.Linear(1569 * 768, 2048)
            self.fc2 = nn.Linear(2048, 51)

        def forward(self, x):
            x = x.view(x.size(0), -1)  # Flatten the input
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Instantiate the model
    two_layer_model = TwoLayerClassifier()

    # Define loss function and optimizer
    two_layer_criterion = nn.CrossEntropyLoss()
    two_layer_optimizer = optim.SGD(two_layer_model.parameters(), lr=0.01)

    # move everything to the GPU
    linear_model = linear_model.to(args_copy.device)
    linear_criterion = linear_criterion.to(args_copy.device)
    two_layer_model = two_layer_model.to(args_copy.device)
    two_layer_criterion = two_layer_criterion.to(args_copy.device)


    for batch_idx, (input_data, target, _, _) in enumerate(data_loader_train):
        empty_mask = torch.zeros((input_data.shape[0], 1568), dtype=torch.bool)
        empty_mask = empty_mask.to(args_copy.device)
        if batch_idx % 10 == 0:
            print(batch_idx)
        input_data = input_data.to(args_copy.device)
        target = target.to(args_copy.device)
        with torch.no_grad():
            features = model.module.forward_encoder(input_data, empty_mask)

        linear_output = linear_model(features)
        linear_loss = linear_criterion(linear_output, target)
        linear_optimizer.zero_grad()
        linear_loss.backward()
        linear_optimizer.step()

        two_layer_output = two_layer_model(features)
        two_layer_loss = two_layer_criterion(two_layer_output, target)
        two_layer_optimizer.zero_grad()
        two_layer_loss.backward()
        two_layer_optimizer.step()

    dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args_copy)
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(args_copy.batch_size),
        num_workers=args_copy.num_workers,
        pin_memory=args_copy.pin_mem,
        drop_last=False
    )

    linear_model.eval()
    two_layer_model.eval()
    correct_linear = 0
    correct_two_layer = 0
    total_samples = 0
    for batch_idx, (input_data, target, _) in enumerate(data_loader_val):
        empty_mask = torch.zeros((input_data.shape[0], 1568), dtype=torch.bool)
        empty_mask = empty_mask.to(args_copy.device)
        if batch_idx % 10 == 0:
            print(batch_idx)
        input_data = input_data.to(args_copy.device)
        target = target.to(args_copy.device)
        with torch.no_grad():
            features = model.module.forward_encoder(input_data, empty_mask)

        linear_output = linear_model(features)
        linear_loss = linear_criterion(linear_output, target)
        _, predicted_linear = torch.max(linear_output.data, 1)
        total_samples += target.size(0)
        correct_linear += (predicted_linear == target).sum().item()

        two_layer_output = two_layer_model(features)
        two_layer_loss = two_layer_criterion(two_layer_output, target)
        _, predicted_two_layer = torch.max(two_layer_output.data, 1)
        correct_two_layer += (predicted_two_layer == target).sum().item()

    accuracy_linear = correct_linear / total_samples
    accuracy_two_layer = correct_two_layer / total_samples
    wandb.log({"linear accuracy": accuracy_linear, "two layer accuracy": accuracy_two_layer})

