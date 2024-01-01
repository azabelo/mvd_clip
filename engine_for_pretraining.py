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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import clip
import matplotlib.pyplot as plt
import seaborn as sns
# END MY CHANGES

Loss_func_choice = {'L1': torch.nn.L1Loss, 'L2': torch.nn.MSELoss, 'SmoothL1': torch.nn.SmoothL1Loss}


def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None, lr_schedule_values=None,
                    wd_schedule_values=None, update_freq=None, time_stride_loss=True, lr_scale=1.0,
                    image_teacher_model=None, video_teacher_model=None, norm_feature=False):

    # MY CHANGES

    # test that the student is the same prior to the start of training
    if epoch % args.knn_freq == 0:
        model.eval()
        with torch.no_grad():
            model.eval()
            empty_mask = torch.zeros((1, 1568), dtype=torch.bool)
            empty_mask = empty_mask.to(args.device)
            ones_features, ones_vid_feats = model(torch.ones((1, 3, 16, 224, 224)).cuda(), empty_mask)
            student_feats = model.module.forward_encoder(torch.ones((1, 3, 16, 224, 224)).cuda(), empty_mask)
            print("student image decoded features (prior to training): ", ones_features[:, 0, :25])
            print("student video decoded features (prior to training): ", ones_vid_feats[:, 0, :25])
            #print("student features (prior to training): ", student_feats[:, 0, :25])
            # if you use cls you should look at the second index to compare with video teacher
            print("student features (prior to training): ", student_feats[:, 1, :25])

            # test that the output of the video teacher doesn't change by passing in a ones vector
            # (found that it doesn't change)
            ones_video_features = video_teacher_model(torch.ones((1, 3, 16, 224, 224)).cuda())
            print("video teacher feats (epoch start): ", ones_video_features[:, 0, :25])
            #only need to do this for one slice (one image)
            ones_image_features = image_teacher_model(torch.ones((1, 3, 224, 224)).cuda())
            print("image teacher feats (epoch start): ", ones_image_features[:, 0, :25])

    # not sure if the pretraining accuracy stuff needs normalization
    if args.knn_freq != -1 and epoch % args.knn_freq == 0:
        pretraining_accuracy(model, video_teacher_model, args)

        # test that the student is the same prior to the start of training
        if epoch % args.knn_freq == 0:
            model.eval()
            with torch.no_grad():
                model.eval()
                empty_mask = torch.zeros((1, 1568), dtype=torch.bool)
                empty_mask = empty_mask.to(args.device)
                ones_features, ones_vid_feats = model(torch.ones((1, 3, 16, 224, 224)).cuda(), empty_mask)
                student_feats = model.module.forward_encoder(torch.ones((1, 3, 16, 224, 224)).cuda(), empty_mask)
                print("student image decoded features (prior to training): ", ones_features[:, 0, :25])
                print("student video decoded features (prior to training): ", ones_vid_feats[:, 0, :25])
                # print("student features (prior to training): ", student_feats[:, 0, :25])
                # if you use cls you should look at the second index to compare with video teacher
                print("student features (prior to training): ", student_feats[:, 1, :25])

                # test that the output of the video teacher doesn't change by passing in a ones vector
                # (found that it doesn't change)
                ones_video_features = video_teacher_model(torch.ones((1, 3, 16, 224, 224)).cuda())
                print("video teacher feats (epoch start): ", ones_video_features[:, 0, :25])
                # only need to do this for one slice (one image)
                ones_image_features = image_teacher_model(torch.ones((1, 3, 224, 224)).cuda())
                print("image teacher feats (epoch start): ", ones_image_features[:, 0, :25])

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

    action_embeddings = torch.load("action_encodings.pth")
    class_names_str = "brush_hair clap draw_sword fall_floor handstand kick pick push run shoot_gun smoke sword turn cartwheel climb dribble fencing hit kick_ball pour pushup shake_hands sit somersault sword_exercise walk catch climb_stairs drink flic_flac hug kiss pullup ride_bike shoot_ball situp stand talk wave chew dive eat golf jump laugh punch ride_horse shoot_bow smile swing_baseball throw"
    action_names = ['brushing hair', 'doing a cartwheel', 'catching', 'chewing', 'clapping', 'climbing',
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
    class_names = class_names_str.split()

    alignment = True
    align_matrix_only = True
    class Alignment_Model(nn.Module):
        def __init__(self):
            super(Alignment_Model, self).__init__()
            clip_model, _ = clip.load("ViT-B/16", device=args.device)
            clip_matrix = clip_model.visual.proj.float()
            # Initialize a linear layer
            self.linear_layer = nn.Linear(768, 512)
            # Set the weight of the linear layer to the CLIP matrix
            with torch.no_grad():
                self.linear_layer.weight.copy_(clip_matrix)

        def forward(self, x):
            empty_mask = torch.zeros((x.shape[0], 1568), dtype=torch.bool)
            empty_mask = empty_mask.to('cuda', non_blocking=True)
            if align_matrix_only:
                model.module.eval()
                with torch.no_grad():
                    model.module.eval()
                    x = model.module.forward_encoder(x, empty_mask)
                    x = x[:, 0, :]
            else:
                x = model.module.forward_encoder(x, empty_mask)
                x = x[:, 0, :]
            x = torch.matmul(x, self.matrix)
            return x

    if alignment:
        alignment_model = Alignment_Model(args.distillation_target_dim, args.distillation_target_dim)
        alignment_model.to(args.device)
        alignment_model.train()
        optimizer = torch.optim.Adam(alignment_model.parameters(), lr=args.lr)
        criterion = nn.MSELoss(reduction='none')

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if not alignment:
            # assign learning rate & weight decay for each step
            update_step = step // update_freq
            it = start_steps + update_step  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None and step % update_freq == 0:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"] * lr_scale
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            videos, videos_for_teacher, bool_masked_pos, class_names = batch
            print(class_names)
            action_names = [action_names[class_names.index(class_name)] for class_name in class_names]
            embeddings = [action_embeddings[action_name] for action_name in action_names]
            embeddings = torch.stack(embeddings)
            print(embeddings.shape)

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
        else:
            # alignment!!!
            videos, videos_for_teacher, bool_masked_pos, class_names = batch
            print(class_names)
            action_names = [action_names[class_names.index(class_name)] for class_name in class_names]
            embeddings = [action_embeddings[action_name] for action_name in action_names]
            embeddings = torch.stack(embeddings)
            videos = videos.to(device, non_blocking=True)

            video_embeddings = alignment_model(videos)
            print(video_embeddings.shape)
            print(embeddings.shape)

            losses = criterion(video_embeddings, embeddings)
            loss = losses.mean(dim=1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log(
                {"batch": step, "alignment loss": loss.mean().item()})




    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def pretraining_accuracy(model, video_teacher_model, args):
    test_teacher = False
    if test_teacher:
        model = video_teacher_model

    # add other finetuning thing here

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
    args_copy.update_freq = 1
    args_copy.batch_size = 8
    args_copy.device = 'cpu'

    dataset_train, args_copy.nb_classes = build_dataset(is_train=True, test_mode=False, args=args_copy)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    if args_copy.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None


    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args_copy.batch_size,
        num_workers=0,
        pin_memory=args_copy.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
    )

    # get the array of text encodings from the text_encodings.pth file
    text_encodings = torch.load('text_encodings.pth')
    print(text_encodings.shape)

    # unnecessary
    linear_probe_video_teacher = False
    if linear_probe_video_teacher:
        class VideoLinearTrainer(nn.Module):
            def __init__(self, classes):
                super(VideoLinearTrainer, self).__init__()
                #just uses class token
                self.linear_layer = nn.Linear(768, classes)

            def forward(self, x):
                # just uses class token
                x = self.linear_layer(x)
                return x

        test_video_teacher = VideoLinearTrainer(51)
        test_video_teacher.to(args_copy.device, non_blocking=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(test_video_teacher.parameters(), lr=1e-3)

        num_epochs = 3

        for epoch in range(num_epochs):
            total_correct = 0
            total_samples = 0
            for batch_idx, (input_data, target, _, _) in enumerate(data_loader_train):
                if batch_idx % 10 == 0:
                    print("vid teacher test: ", batch_idx)
                input_data = input_data.to('cuda', non_blocking=True)
                target = target.to('cuda', non_blocking=True)
                # just uses class token
                features = video_teacher_model(input_data)[:,0,:]
                output = test_video_teacher(features)
                loss = criterion(output, target)

                # Compute accuracy
                predicted_labels = torch.argmax(output, dim=1)
                correct = (predicted_labels == target).sum().item()
                total_correct += correct
                total_samples += target.size(0)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Calculate and print accuracy
            accuracy = total_correct / total_samples
            print(f"Accuracy: {accuracy * 100:.2f}%")
    # end unnecessary


    class LinearClassifier(nn.Module):
        def __init__(self):
            super(LinearClassifier, self).__init__()

            if test_teacher:
                self.fc = nn.Linear(768 * 1568, 51)
            else:
                self.fc = nn.Linear(768*1569, 51)

        def forward(self, x):
            # when for video teacher:
            x = x.reshape(x.size(0), -1)

            #x = x.view(x.size(0), -1)  # Flatten the input

            x = self.fc(x)
            return x

    # Instantiate the model
    linear_model = LinearClassifier()
    linear_criterion = nn.CrossEntropyLoss()
    linear_optimizer = optim.SGD(linear_model.parameters(), lr=1e-3)




    # class TwoLayerClassifier(nn.Module):
    #     def __init__(self):
    #         super(TwoLayerClassifier, self).__init__()
    #         self.fc1 = nn.Linear(1569 * 768, 2048)
    #         self.fc2 = nn.Linear(2048, 51)
    #
    #     def forward(self, x):
    #         x = x.view(x.size(0), -1)  # Flatten the input
    #         x = torch.relu(self.fc1(x))
    #         x = self.fc2(x)
    #         return x
    #
    # # Instantiate the model
    # two_layer_model = TwoLayerClassifier()
    #
    # # Define loss function and optimizer
    # two_layer_criterion = nn.CrossEntropyLoss()
    # two_layer_optimizer = optim.SGD(two_layer_model.parameters(), lr=2e-5)

    # move everything to the GPU
    linear_model = linear_model.to('cuda')
    linear_criterion = linear_criterion.to('cuda')
    linear_model.train()
    # two_layer_model = two_layer_model.to(args_copy.device)
    # two_layer_criterion = two_layer_criterion.to(args_copy.device)

    knn_classifier19 = KNeighborsClassifier(n_neighbors=19, algorithm='brute', metric='cosine')
    knn_classifier5 = KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='cosine')

    knn_features_train = np.empty((0, 768))
    knn_labels_train = np.empty(0)

    zero_shot_correct = 0
    total_zero_shot = 0

    video_encodings = []
    image_encodings = []
    for batch_idx, (input_data, target, _, _) in enumerate(data_loader_train):
        linear_optimizer.zero_grad()

        empty_mask = torch.zeros((input_data.shape[0], 1568), dtype=torch.bool)
        empty_mask = empty_mask.to('cuda', non_blocking=True)
        if batch_idx % 10 == 0:
            print("knn train: ", batch_idx)
        input_data = input_data.to('cuda', non_blocking=True)
        target = target.to('cuda', non_blocking=True)

        model.eval()
        with torch.no_grad():
            model.eval()
            # if you want to test video teacher:
            if test_teacher:
                # video mae v2
                # features = model.forward_features(input_data)
                # video teacher
                features = model.forward(input_data)
            else:
                features = model.module.forward_encoder(input_data, empty_mask)
                # image_features, _ = model.module.forward(input_data, empty_mask)

            # features = features.detach()
            cls_token = features[:, 0, :]
            # first_token = image_features[:, 0, :]

            knn_features_train = np.append(knn_features_train, cls_token.cpu().numpy(), axis=0)
            knn_labels_train = np.append(knn_labels_train, target.cpu().numpy(), axis=0)
            # knn_features_train = np.concatenate((knn_features_train, cls_token.cpu().numpy()), axis=0)
            # knn_labels_train = np.concatenate((knn_labels_train, target.cpu().numpy()), axis=0)


            # naive zero shot testing
            clip_model, preprocess = clip.load("ViT-B/16", device=args.device)
            # multiply the features by the model.visual.proj matrix (not to be done when model is the teacher)
            #vid_space_features = torch.matmul(cls_token, clip_model.visual.proj.float())
            vid_space_features = cls_token

            vid_space_features /= vid_space_features.norm(dim=-1, keepdim=True)
            video_encodings.append(vid_space_features)

            #img_space_features = torch.matmul(first_token, clip_model.visual.proj.float())
            # img_space_features = first_token
            #
            # img_space_features /= img_space_features.norm(dim=-1, keepdim=True)
            # image_encodings.append(img_space_features)

            # for each of the features, find the cosine similarity with each of the text features
            # tensor1 = vid_space_features.unsqueeze(1)
            # tensor2 = text_encodings.unsqueeze(0)
            # vid_cosine_sim = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=2)
            #
            # total_zero_shot += vid_cosine_sim.shape[0]
            # # find the index of the highest cosine similarity for each of the features
            # max_index = torch.argmax(vid_cosine_sim, dim=1)
            # print("max index: ", max_index)
            # max_index = max_index // 48
            # print("max index: ", max_index)
            # print("target: ", target)
            # # find how many of these match the target
            # zero_shot_correct += torch.sum(max_index == target).item()
            # zero_shot_accuracy = zero_shot_correct / total_zero_shot
            # print("zero shot accuracy: ", zero_shot_accuracy)

            # # find the sum of every 48 elements in the cosine sim
            # cosine_sim = cosine_sim.view(8, 51, 48)
            # cosine_sim = torch.sum(cosine_sim, dim=2)
            # # find the index of the highest cosine similarity for each of the features
            # max_index = torch.argmax(cosine_sim, dim=1)
            # print("max index: ", max_index)

        linear_output = linear_model(features)
        linear_loss = linear_criterion(linear_output, target)
        linear_loss.backward()
        linear_optimizer.step()

        # two_layer_output = two_layer_model(features)
        # two_layer_loss = two_layer_criterion(two_layer_output, target)
        # two_layer_optimizer.zero_grad()
        # two_layer_loss.backward()
        # two_layer_optimizer.step()

        linear_predictions = linear_output.argmax(dim=1)
        linear_accuracy = (linear_predictions == target).float().mean().item()
        # two_layer_predictions = two_layer_output.argmax(dim=1)
        # two_layer_accuracy = (two_layer_predictions == target).float().mean().item()

        # send a tensor of ones through the linear model
        # to make sure the model is not broken
        # ones = torch.ones((input_data.shape[0], 768))
        # ones = ones.to('cuda', non_blocking=True)
        # ones_output = linear_model(ones)
        # print("ones output: ", ones_output)


        # print("linear loss: ", linear_loss.item())
        # print("linear accuracy: ", linear_accuracy)
        wandb.log({'linear_loss': linear_loss.item(),
                   'linear_accuracy train': linear_accuracy})

        # wandb.log({'linear_loss': linear_loss.item(),
        #            'two_layer_loss': two_layer_loss.item(),
        #           'linear_accuracy train': linear_accuracy,
        #         'two_layer_accuracy train': two_layer_accuracy})


    ## this is for generating the heatmap of the cosine similarities of the videos
    video_encodings = torch.cat(video_encodings)
    vid_path = "vid_encodings_no_matrix.pth"
    torch.save(video_encodings, vid_path)
    print(f"Text encodings saved to {vid_path}")

    # image_encodings = torch.cat(image_encodings)
    # img_path = "img_encodings_no_matrix.pth"
    # torch.save(image_encodings, img_path)
    # print(f"Text encodings saved to {img_path}")


    knn_classifier19.fit(knn_features_train, knn_labels_train)
    knn_classifier5.fit(knn_features_train, knn_labels_train)

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

    model.eval()
    linear_model.eval()
    with torch.no_grad():
        model.eval()
        linear_model.eval()
        # two_layer_model.eval()
        correct_linear = 0
        correct_two_layer = 0
        total_samples = 0

        knn_features_val = np.empty((0, 768))
        knn_labels_val = np.empty(0)
        for batch_idx, (input_data, target, _) in enumerate(data_loader_val):
            empty_mask = torch.zeros((input_data.shape[0], 1568), dtype=torch.bool)
            empty_mask = empty_mask.to('cuda', non_blocking=True)
            if batch_idx % 10 == 0:
                print("knn val: ", batch_idx)
            input_data = input_data.to('cuda', non_blocking=True)
            target = target.to('cuda', non_blocking=True)

            # if you want to test video teacher:
            if test_teacher:
                # video mae v2
                #features = model.forward_features(input_data)
                # video teacher
                features = model.forward(input_data)
            else:
                features = model.module.forward_encoder(input_data, empty_mask)

            # features = features.detach()
            cls_token = features[:, 0, :]
            knn_features_val = np.append(knn_features_val ,cls_token.cpu().numpy(), axis=0)
            knn_labels_val = np.append(knn_labels_val, target.cpu().numpy(), axis=0)
            # knn_features_val = np.concatenate((knn_features_val, cls_token.cpu().numpy()), axis=0)
            # knn_labels_val = np.concatenate((knn_labels_val, target.cpu().numpy()), axis=0)

            total_samples += target.size(0)

            linear_output = linear_model(features)
            linear_loss = linear_criterion(linear_output, target)
            _, predicted_linear = torch.max(linear_output.data, 1)
            correct_linear += (predicted_linear == target).sum().item()

            # two_layer_output = two_layer_model(features)
            # two_layer_loss = two_layer_criterion(two_layer_output, target)
            # _, predicted_two_layer = torch.max(two_layer_output.data, 1)
            # correct_two_layer += (predicted_two_layer == target).sum().item()

    val_predictions19 = knn_classifier19.predict(knn_features_val)
    val_predictions5 = knn_classifier5.predict(knn_features_val)
    print("preds")
    print(val_predictions5)
    correct_knn19 = (val_predictions19 == knn_labels_val).sum()
    correct_knn5 = (val_predictions5 == knn_labels_val).sum()
    accuracy_knn19 = correct_knn19 / total_samples
    accuracy_knn5 = correct_knn5 / total_samples
    print("knn correct 5: ", correct_knn5)
    print("knn correct 19: ", correct_knn19)

    accuracy_linear = None
    accuracy_two_layer = None
    accuracy_linear = correct_linear / total_samples
    # accuracy_two_layer = correct_two_layer / total_samples
    wandb.log({"linear accuracy": accuracy_linear,
               "two layer accuracy": accuracy_two_layer,
               "knn accuracy 19": accuracy_knn19,
               "knn accuracy 5": accuracy_knn5})
    del knn_features_train
    del knn_labels_train
    del knn_features_val
    del knn_labels_val
    del dataset_val
    del sampler_val
    del data_loader_val
    del linear_model
    # del two_layer_model
    del linear_optimizer
    # del two_layer_optimizer
    del linear_criterion
    # del two_layer_criterion
    torch.cuda.empty_cache()

