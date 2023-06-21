# Copyright (c) Facebook, Inc. and its affiliates.
import pickle

import torch
import datetime
import logging
import math
import time
import sys, os

from torch.distributed.distributed_c10d import reduce
from utils.ap_calculator import APCalculator
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
)
torch.autograd.set_detect_anomaly(True)
import wandb


def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch(
    args,
    curr_epoch,
    model,
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
):
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5, 0.6, 0.7, 0.8, 0.9],
        class2type_map=dataset_config.class2type,
        exact_eval=False,
    )

    curr_iter = curr_epoch * len(dataset_loader)  # 0  #
    max_iters = args.max_epoch * len(dataset_loader)  # len(dataset_loader)  #
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        # curr_lr = args.base_lr
        for key in batch_data_label:
            if key in ['scan_path', 'intrinsic_matrix', 'cam_to_world', 'nerf_ckpt_path', 'corner_bboxes_path']:
                continue
            batch_data_label[key] = batch_data_label[key].to(net_device)
            if dataset_config.pseudo_batch_size > 1:
                if key in ["gt_box_corners"]:
                    sh = batch_data_label[key].shape
                    batch_data_label[key] = batch_data_label[key].reshape(-1, sh[2], sh[3], sh[4])
                if key in ['gt_box_sem_cls_label', 'gt_box_present', 'gt_box_angles']:
                    batch_data_label[key] = batch_data_label[key].reshape(-1, 1)

        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
            'nerf_ckpt_path': batch_data_label["nerf_ckpt_path"],
            'gt_box_corners': batch_data_label["gt_box_corners"],
            'gt_box_sem_cls_label': batch_data_label["gt_box_sem_cls_label"],
            'gt_box_present': batch_data_label['gt_box_present'],
            'gt_box_angles': batch_data_label['gt_box_angles'],
            'scan_path': batch_data_label['scan_path'],
        }
        outputs, return_gt = model(inputs, net_device=net_device)
        batch_data_label.update(return_gt)
        # Compute loss
        loss, loss_dict = criterion(outputs, batch_data_label)
        if isinstance(outputs, list):
            outputs = outputs[0]

        if args.save_pred and False:
            save_outputs = {
                'sem_cls_logits': outputs['outputs']['sem_cls_logits'].cpu(),
                'box_corners': outputs['outputs']['box_corners'].cpu(),
            }
            save_batch_data_label = {
                'gt_box_corners': batch_data_label['gt_box_corners'].cpu(),
                'gt_box_sem_cls_label': batch_data_label['gt_box_sem_cls_label'].cpu(),
                'scan_idx': batch_data_label['scan_idx'],
                'scan_path': batch_data_label['scan_path'],
                'nerf_ckpt_path': batch_data_label['nerf_ckpt_path'],
                'intrinsic_matrix': batch_data_label['intrinsic_matrix'],
                'cam_to_world': batch_data_label['cam_to_world'],
            }
            if args.test_only: # or True:
                save_batch_data_label['nerf_rgbs'] = batch_data_label['nerf_rgbs']
                if 'nerf_depth' in batch_data_label:
                    save_batch_data_label['nerf_depth'] = batch_data_label['nerf_depth']
                if 'nerf_depth_vis' in batch_data_label:
                    save_batch_data_label['nerf_depth_vis'] = batch_data_label['nerf_depth_vis']
                if args.arch_type in ['coarse_fine', 'coarse',]:
                    save_batch_data_label['coarse_nerf_rgbs'] = batch_data_label['coarse_nerf_rgbs']
                    if 'coarse_nerf_depth' in batch_data_label:
                        save_batch_data_label['coarse_nerf_depth'] = batch_data_label['coarse_nerf_depth']
                    if 'coarse_nerf_depth_vis' in batch_data_label:
                        save_batch_data_label['coarse_nerf_depth_vis'] = batch_data_label['coarse_nerf_depth_vis']
            save_dict = {
                'outputs': save_outputs,
                'label': save_batch_data_label,
            }
            with open(os.path.join(args.log_dir, 'train_{:03d}_{:04d}.pkl'.format(curr_epoch, batch_idx)), 'wb') as f:
                pickle.dump(save_dict, f)
        loss_reduced = all_reduce_average(loss)
        loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)

        loss.backward()
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        if curr_iter % args.log_metrics_every == 0:
            # This step is slow. AP is computed approximately and locally during training.
            # It will gather outputs and ground truth across all ranks.
            # It is memory intensive as point_cloud ground truth is a large tensor.
            # If GPU memory is not an issue, uncomment the following lines.
            # outputs["outputs"] = all_gather_dict(outputs["outputs"])
            # batch_data_label = all_gather_dict(batch_data_label)
            ap_calculator.step_meter(outputs, batch_data_label)

        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced.item())

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict.update(loss_dict)
            train_dict["batch_time"] = time_delta.avg
            logger.log_scalars(train_dict, curr_iter, prefix="Train/")
        curr_iter += 1
        barrier()

    if args.USE_WANDB:
        wandb_train_dict = {}
        for i, (k, v) in enumerate(train_dict.items()):
            wandb_train_dict['train_' + k] = v
        wandb.log(wandb_train_dict, step=curr_epoch)

    return ap_calculator


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
):

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        class2type_map=dataset_config.class2type,
        exact_eval=False,
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    # model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        for key in batch_data_label:
            if key in ['scan_path', 'intrinsic_matrix', 'cam_to_world', 'nerf_ckpt_path', 'corner_bboxes_path']:
                continue
            batch_data_label[key] = batch_data_label[key].to(net_device)
            if dataset_config.pseudo_batch_size > 1:
                if key in ["gt_box_corners"]:
                    sh = batch_data_label[key].shape
                    batch_data_label[key] = batch_data_label[key].reshape(-1, sh[2], sh[3], sh[4])
                if key in ['gt_box_sem_cls_label', 'gt_box_present', 'gt_box_angles']:
                    batch_data_label[key] = batch_data_label[key].reshape(-1, 1)

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
            'nerf_ckpt_path': batch_data_label["nerf_ckpt_path"],
            'gt_box_corners': batch_data_label["gt_box_corners"],
            'gt_box_sem_cls_label': batch_data_label["gt_box_sem_cls_label"],
            'gt_box_present': batch_data_label['gt_box_present'],
            'gt_box_angles': batch_data_label['gt_box_angles'],
            'scan_path': batch_data_label['scan_path'],
        }
        outputs, return_gt = model(inputs, net_device=net_device)
        batch_data_label.update(return_gt)
        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"
        if isinstance(outputs, list):
            outputs = outputs[0]

        if args.save_pred:
            save_outputs = {
                'sem_cls_logits': outputs['outputs']['sem_cls_logits'],
                'box_corners': outputs['outputs']['box_corners'],
            }
            save_batch_data_label = {
                'gt_box_corners': batch_data_label['gt_box_corners'],
                'gt_box_sem_cls_label': batch_data_label['gt_box_sem_cls_label'],
                'scan_idx': batch_data_label['scan_idx'],
                'scan_path': batch_data_label['scan_path'],
                'intrinsic_matrix': batch_data_label['intrinsic_matrix'],
                'cam_to_world': batch_data_label['cam_to_world'],
            }
            if args.test_only:
                save_batch_data_label['nerf_rgbs'] = batch_data_label['nerf_rgbs']
                if 'nerf_depth' in batch_data_label:
                    save_batch_data_label['nerf_depth'] = batch_data_label['nerf_depth']
                if 'nerf_depth_vis' in batch_data_label:
                    save_batch_data_label['nerf_depth_vis'] = batch_data_label['nerf_depth_vis']
                if args.arch_type in ['coarse_fine', 'coarse',]:
                    save_batch_data_label['coarse_nerf_rgbs'] = batch_data_label['coarse_nerf_rgbs']
                    if 'coarse_nerf_depth' in batch_data_label:
                        save_batch_data_label['coarse_nerf_depth'] = batch_data_label['coarse_nerf_depth']
                    if 'coarse_nerf_depth_vis' in batch_data_label:
                        save_batch_data_label['coarse_nerf_depth_vis'] = batch_data_label['coarse_nerf_depth_vis']
            save_dict = {
                'outputs': save_outputs,
                'label': save_batch_data_label,
            }
            with open(os.path.join(args.log_dir, 'eval_{:03d}_{:04d}.pkl'.format(curr_epoch, batch_idx)), 'wb') as f:
                pickle.dump(save_dict, f)
        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)
        ap_calculator.step_meter(outputs, batch_data_label)
        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg
                test_dict.update(loss_dict)
        curr_iter += 1
        barrier()
    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")
    if args.USE_WANDB:
        wandb_test_dict = {}
        for i, (k, v) in enumerate(test_dict.items()):
            wandb_test_dict['test_' + k] = v
        wandb.log(wandb_test_dict, step=curr_epoch)
    return ap_calculator
