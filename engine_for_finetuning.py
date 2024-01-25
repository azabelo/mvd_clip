import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
from einops import rearrange
from torch.utils.data._utils.collate import default_collate
import torch.nn.functional as F

# MY CHANGES
import wandb
import random
import matplotlib.pyplot as plt
# END MY CHANGES


def train_class_batch(model, samples, target, criterion):

    outputs = model(samples)
    loss = criterion(outputs, target)
    # MY CHANGES
    # print("targets")
    # print(target)
    # print("outputs")
    # print(outputs)
    # END MY CHANGES

    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn=None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
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
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        # MY CHANGES
        wandb.log({"epoch": epoch, "batch": step, "train_loss": loss_value, "max_lr": max_lr, "min_lr": min_lr,
                   "weight_decay": weight_decay_value, "grad_norm": grad_norm, "loss_scale": loss_scale_value,
                   "class_acc": class_acc})
        # END MY CHANGES

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # MY CHANGES
    wandb.log({"val_acc (top 1)": metric_logger.acc1.global_avg, "val_acc (top 5)": metric_logger.acc5.global_avg, "val_loss": metric_logger.loss.global_avg})
    # END MY CHANGES

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i].replace('[', '').replace(']', ''), \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100, final_top5*100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]



def align_class_batch(model, samples, text, criterion):

    loss, vid_pred_correct, text_pred_correct = model(samples, text)
    print("vid correct / 8:", vid_pred_correct)
    print("text correct / 8:", text_pred_correct)

    # outputs
    return loss, vid_pred_correct, text_pred_correct

def align_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn=None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        class_names_str = "brush_hair clap draw_sword fall_floor handstand kick pick push run shoot_gun smoke sword turn cartwheel climb dribble fencing hit kick_ball pour pushup shake_hands sit somersault sword_exercise walk catch climb_stairs drink flic_flac hug kiss pullup ride_bike shoot_ball situp stand talk wave chew dive eat golf jump laugh punch ride_horse shoot_bow smile swing_baseball throw"
        all_action_names = ['brushing hair', 'doing a cartwheel', 'catching', 'chewing', 'clapping', 'climbing',
                            'climbing stairs', 'diving', 'drawing a sword', 'dribbling', 'drinking', 'eating',
                            'falling to the floor', 'fencing', 'doing flic flac', 'golfing', 'doing a handstand',
                            'hitting',
                            'hugging', 'jumping', 'kicking', 'kicking a ball', 'kissing', 'laughing', 'picking',
                            'pouring',
                            'doing pullups', 'punching', 'pushing', 'doing pushups', 'riding a bike',
                            'riding a horse',
                            'running', 'shaking hands', 'shooting a ball', 'shooting a bow', 'shooting a gun',
                            'sitting',
                            'doing situps', 'smiling', 'smoking', 'doing a somersault', 'standing',
                            'swinging a baseball bat',
                            'using a sword', 'doing sword exercises', 'talking', 'throwing', 'turning', 'walking',
                            'waving']
        all_class_names = class_names_str.split()
        templates = [
            'a photo of a person {}.',
            'a video of a person {}.',
            'a example of a person {}.',
            'a demonstration of a person {}.']

        print("_____________________")
        print(targets)
        # repeat the training loop for each template
        for template in templates:
            text = [template.format(all_action_names[targets[i]]) for i in range(len(targets))]

            # if mixup_fn is not None:
            #     samples, targets = mixup_fn(samples, targets)

            if loss_scaler is None:
                samples = samples.half()
                loss, vid_pred_correct, text_pred_correct = align_class_batch(
                    model, samples, text, criterion)
            else:
                with torch.cuda.amp.autocast():
                    loss, vid_pred_correct, text_pred_correct = align_class_batch(
                        model, samples, text, criterion)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            if loss_scaler is None:
                loss /= update_freq
                model.backward(loss)
                model.step()

                if (data_iter_step + 1) % update_freq == 0:
                    # model.zero_grad()
                    # Deepspeed will call step() & model.zero_grad() automatic
                    if model_ema is not None:
                        model_ema.update(model)
                grad_norm = None
                loss_scale_value = get_loss_scale_for_deepspeed(model)
            else:
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss /= update_freq
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order,
                                        update_grad=(data_iter_step + 1) % update_freq == 0)
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()

            class_acc = (vid_pred_correct + text_pred_correct) / (2 * len(targets))
            # if mixup_fn is None:
            #     class_acc = (output.max(-1)[-1] == targets).float().mean()
            # else:
            #     class_acc = None

            metric_logger.update(loss=loss_value)
            metric_logger.update(class_acc=class_acc)
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
                log_writer.update(class_acc=class_acc, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()

            # MY CHANGES
            wandb.log({"epoch": epoch, "batch": step, "train_loss": loss_value, "max_lr": max_lr, "min_lr": min_lr,
                       "weight_decay": weight_decay_value, "grad_norm": grad_norm, "loss_scale": loss_scale_value,
                       "class_acc": class_acc})
            # END MY CHANGES

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def efficient_align_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn=None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    train_video_embeddings=None, train_targets=None, text_encodings=None, batch_size=64,
                              linear_model=None, linear_criterion=None,
                              linear_optimizer=None,
                              linear_loss_scaler=None, linear_model_ema=None,
                              ):

    model.train(True)
    linear_model.train(True)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    # scramble the tensor
    permutation = torch.randperm(train_video_embeddings.shape[0])
    random_train_video_embeddings = train_video_embeddings[permutation]
    random_train_targets = train_targets[permutation]

    # group in batches
    num_batches = random_train_video_embeddings.shape[0] // batch_size
    random_train_video_embeddings = random_train_video_embeddings[:num_batches * batch_size]
    random_train_targets = random_train_targets[:num_batches * batch_size]
    random_train_video_embeddings = random_train_video_embeddings.reshape(num_batches, batch_size, -1)
    random_train_targets = random_train_targets.reshape(num_batches, batch_size)

    total_linear_correct = 0
    total_vid_correct = 0
    total_text_correct = 0
    batched_data = [(i,j) for _, (i,j) in enumerate(zip(random_train_video_embeddings, random_train_targets))]
    batch_count = 0
    for data_iter_step, (samples, targets, _, _) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):

        # so janky lmao
        video_embeddings, targets = batched_data[batch_count]
        video_embeddings = video_embeddings.half().to(device)
        batch_count += 1

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        # prompt_index = random.randint(0, 47)
        # text_embeddings = text_encodings[torch.tensor([48 * class_index + prompt_index for class_index in targets])]
        text_embeddings = text_encodings[targets]
        text_embeddings = text_embeddings.to(device)
        video_embeddings = video_embeddings.to(device)

        if loss_scaler is None:
            samples = samples.half()
            loss, vid_preds_correct, text_preds_correct = model(video_embeddings, text_embeddings)
            total_vid_correct += vid_preds_correct
            total_text_correct += text_preds_correct
        else:
            with torch.cuda.amp.autocast():
                loss, vid_preds_correct, text_preds_correct = model(video_embeddings, text_embeddings)
                total_vid_correct += vid_preds_correct
                total_text_correct += text_preds_correct

        loss_value = loss.item()

        # note that the linear model is not affected by anything like loss scaling or gradient accumulation

        linear_optimizer.zero_grad()
        linear_logits = linear_model(video_embeddings)
        print(linear_logits)
        print(targets)
        linear_loss = linear_criterion(linear_logits.cuda(), targets.cuda())
        linear_loss.backward()
        print("linear loss: ", linear_loss)
        linear_optimizer.step()
        # probabilities = torch.nn.functional.softmax(linear_logits, dim=1).cuda()
        predictions = torch.argmax(linear_logits, dim=1).cuda()
        linear_correct = (predictions.cuda() == targets.cuda()).sum().item()
        total_linear_correct += linear_correct

        linear_logits = linear_model(video_embeddings)
        # probabilities = torch.nn.functional.softmax(linear_logits, dim=1).cuda()
        predictions = torch.argmax(linear_logits, dim=1).cuda()
        linear_correct = (predictions.cuda() == targets.cuda()).sum().item()
        total_linear_correct += linear_correct
        total_linear_loss += linear_criterion(linear_logits.cuda(), targets.cuda())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        class_acc = text_preds_correct
        # if mixup_fn is None:
        #     class_acc = (output.max(-1)[-1] == targets).float().mean()
        # else:
        #     class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
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
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        # MY CHANGES
        wandb.log({"epoch": epoch, "batch": step, "train_loss": loss_value, "max_lr": max_lr, "min_lr": min_lr,
                   "weight_decay": weight_decay_value, "grad_norm": grad_norm, "loss_scale": loss_scale_value,
                   "text_correct": text_preds_correct, "vid_correct": vid_preds_correct,
                   "linear_loss": linear_loss, "linear_acc": linear_correct})
        # END MY CHANGES

    wandb.log({"total_linear_correct": total_linear_correct, "total_vid_correct": total_vid_correct,
               "total_text_correct": total_text_correct})


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def cls_token_similarity(model: torch.nn.Module,
                    device: torch.device,
                    test_video_embeddings=None, test_targets=None, text_encodings=None, batch_size=64):

    model.eval()
    with torch.no_grad():
        model.eval()

        # group in batches
        num_batches = test_video_embeddings.shape[0] // batch_size
        test_video_embeddings = test_video_embeddings[:num_batches * batch_size]
        test_targets = test_targets[:num_batches * batch_size]
        test_video_embeddings = test_video_embeddings.reshape(num_batches, batch_size, -1)
        test_targets = test_targets.reshape(num_batches, batch_size)


        batched_data = [(i,j) for _, (i,j) in enumerate(zip(test_video_embeddings, test_targets))]
        batch_count = 0
        # total_loss = 0
        # total_vid_preds_correct = 0
        # total_text_preds_correct = 0
        # total_examples = 0
        # total_class_correct = 0

        cls_tokens = []
        while batch_count < len(batched_data):
            print("batch count: ", batch_count)
            video_embeddings, targets = batched_data[batch_count]
            video_embeddings = video_embeddings.half().to(device)
            targets.to(device)
            batch_count += 1
            # total_examples += video_embeddings.shape[0]


            # clip-style val prediction loss
            video_vectors = model.module.get_video_embeddings(video_embeddings)
            cls_token = video_vectors #[:, 0, :]
            cls_token = cls_token / torch.norm( cls_token, dim=1, keepdim=True)
            cls_tokens.append(cls_token)

        cls_tokens = torch.cat(cls_tokens, dim=0)
        cls_token_similarity = torch.matmul(cls_tokens, cls_tokens.T)
        log_matrix(cls_token_similarity, "cls_token_similarity", 400)
    return

def align_val_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn=None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    test_video_embeddings=None, test_targets=None, text_encodings=None, batch_size=64,
                        linear_model=None, linear_criterion=None,
                        ):

    # val scrambles the order, so we call this method to visualize the ordered cls token similarity
    cls_token_similarity(model=model, test_video_embeddings=test_video_embeddings, test_targets=test_targets,
                         text_encodings=text_encodings, device=device, batch_size=batch_size)

    linear_model.eval()
    model.eval()
    with torch.no_grad():
        model.eval()
        linear_model.eval()


        permutation = torch.randperm(test_video_embeddings.shape[0])
        test_video_embeddings = test_video_embeddings[permutation]
        test_targets = test_targets[permutation]

        # group in batches
        num_batches = test_video_embeddings.shape[0] // batch_size
        test_video_embeddings = test_video_embeddings[:num_batches * batch_size]
        test_targets = test_targets[:num_batches * batch_size]
        test_video_embeddings = test_video_embeddings.reshape(num_batches, batch_size, -1)
        test_targets = test_targets.reshape(num_batches, batch_size)


        batched_data = [(i,j) for _, (i,j) in enumerate(zip(test_video_embeddings, test_targets))]
        batch_count = 0
        total_loss = 0
        total_vid_preds_correct = 0
        total_text_preds_correct = 0
        total_examples = 0
        total_class_correct = 0

        total_linear_correct = 0
        total_linear_loss = 0

        while batch_count < len(batched_data):
            print("batch count: ", batch_count)
            video_embeddings, targets = batched_data[batch_count]
            video_embeddings = video_embeddings.half().to(device)
            targets.to(device)
            batch_count += 1
            total_examples += video_embeddings.shape[0]


            linear_logits = linear_model(video_embeddings)
            # probabilities = torch.nn.functional.softmax(linear_logits, dim=1).cuda()
            predictions = torch.argmax(linear_logits, dim=1).cuda()
            linear_correct = (predictions.cuda() == targets.cuda()).sum().item()
            total_linear_correct += linear_correct
            total_linear_loss += linear_criterion(linear_logits.cuda(), targets.cuda())


            # clip-style val prediction loss
            video_vectors = model.module.get_video_embeddings(video_embeddings)
            cls_token = video_vectors #[:, 0, :]
            cls_token = cls_token / torch.norm( cls_token, dim=-1, keepdim=True)

            text_encodings.to(device)
            # text_encodings_mean = text_encodings.mean(dim=1, keepdim=True)
            # text_encodings_std = text_encodings.std(dim=1, keepdim=True)
            # text_encodings = (text_encodings - text_encodings_mean) / text_encodings_std

            print("video vectors shape: ", video_vectors.shape)
            print("text encodings shape: ", text_encodings.shape)
            probs= (text_encodings @ video_vectors.T) / model.module.logit_scale
            # take the softmax over the text encodings
            print("probs shape: ", probs.shape)
            # sum each group of 48
            #probs = probs.reshape(batch_size, 48, -1).sum(dim=1)
            #print("sims shape: ", probs.shape)
            print(probs)
            # take the argmax
            class_preds = torch.argmax(probs, dim=0).to(device)
            print(class_preds)
            print(targets)
            # compute accuracy

            class_preds = class_preds.to(device)
            targets = targets.to(device)
            class_correct = (class_preds == targets).sum().item()
            total_class_correct += class_correct


            # training-style val loss

            # prompt_index = random.randint(0, 47)
            text_embeddings = text_encodings[targets]
            text_embeddings = text_embeddings.to(device)
            video_embeddings = video_embeddings.to(device)

            loss, vid_preds_correct, text_preds_correct = model.module(video_embeddings, text_embeddings)
            loss_value = loss.item()
            total_loss += loss_value * video_embeddings.shape[0]
            total_vid_preds_correct += vid_preds_correct
            total_text_preds_correct += text_preds_correct

        avg_loss = total_loss / total_examples
        # avg_vid_acc = total_vid_preds_correct / total_examples
        # avg_text_acc = total_text_preds_correct / total_examples
        # avg_class_acc = total_class_correct / total_examples

        avg_linear_loss = total_linear_loss / total_examples
        linear_acc = total_linear_correct / total_examples

        # MY CHANGES
        wandb.log({"epoch": epoch, "val_loss": avg_loss, "val_vid_acc": total_vid_preds_correct,
                   "val_text_acc": total_text_preds_correct, "val_class_acc": total_class_correct,
                   "val_linear_loss": avg_linear_loss, "val_linear_acc": linear_acc})
        # END MY CHANGES

    return


def log_matrix(matrix, title, dpi):

    matrix = matrix.clone().detach().cpu().numpy()
    # Create a figure and axis
    fig, ax = plt.subplots(dpi=dpi)
    # Plot the heatmap
    cax = ax.matshow(matrix, cmap='viridis')
    # Set the aspect ratio to be equal
    ax.set_aspect('equal')
    # Add a colorbar
    fig.colorbar(cax)
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    # Log the figure to WandB
    wandb.log({title: wandb.Image(fig)})
    plt.close(fig)