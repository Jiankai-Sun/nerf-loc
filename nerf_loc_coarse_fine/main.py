# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os, time
import sys
import pickle
import random

import numpy as np
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler

# 3DETR codebase specific imports
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from optimizer import build_optimizer
from criterion import build_criterion
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier
from utils.misc import my_worker_init_fn
from utils.io import save_checkpoint, resume_if_possible
from utils.logger import Logger
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
import wandb
os.environ["WANDB_API_KEY"] = "b00b2711e75723b6df804b383842ad17c46a84b0"

module_path = os.path.abspath(os.path.join('..', 'nerf-pytorch'))
if module_path not in sys.path:
    sys.path.append(module_path)
from utils.nerf_utils import create_nerf_model

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int, help='9')
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=1, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0., type=float, help='0.1')
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=1, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0., type=float, help='0.1')
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.0, type=float, help='0.3')
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=1, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")

    ##### Set Loss #####
    ### Matcher
    parser.add_argument("--matcher_giou_cost", default=2, type=float)
    parser.add_argument("--matcher_cls_cost", default=1, type=float)
    parser.add_argument("--matcher_center_cost", default=0, type=float)
    parser.add_argument("--matcher_objectness_cost", default=0, type=float)

    ### Loss Weights
    parser.add_argument("--loss_giou_weight", default=0, type=float)
    parser.add_argument("--loss_sem_cls_weight", default=1, type=float)
    parser.add_argument("--loss_coord_weight", default=1, type=float)
    parser.add_argument(
        "--loss_no_object_weight", default=0.2, type=float
    )  # "no object" or "background" class for detection
    parser.add_argument("--loss_angle_cls_weight", default=0.1, type=float)
    parser.add_argument("--loss_angle_reg_weight", default=0.5, type=float)
    parser.add_argument("--loss_center_weight", default=5.0, type=float)
    parser.add_argument("--loss_size_weight", default=1.0, type=float)

    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", required=True, type=str, choices=["scannet", "sunrgbd", "nerf"]
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default='../objectron_dataset/',
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used,"
             "../objectron_dataset/train/bike/batch-0/10"
             "../nerf-pytorch/logs/objectron_small/renderonly_path_199999/",
    )
    parser.add_argument(
        "--meta_data_dir",
        type=str,
        default=None,
        help="Root directory containing the metadata files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=1, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=500, type=int, help='720, ')
    parser.add_argument("--eval_every_epoch", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--test_ckpt", default=None, type=str)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--log_every", default=1, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=100, type=int)

    ##### Distributed Training #####
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)
    parser.add_argument("--input_type", default="nerf", type=str, help='rgb / nerf / rendered_rgb / rendered_depth / nerf_rendered_rgb / nerf_rendered_depth / nerf_rendered_rgb_rendered_depth / rendered_rgb_rendered_depth')
    parser.add_argument("--version", default="v2_corner", type=str, help='v1_csa / v2_corner')
    parser.add_argument('--USE_WANDB', type=str2bool, default=False, help="use WANDB")
    parser.add_argument("--save_pred", type=str2bool, default=True)
    parser.add_argument("--model_type", default="3detr", type=str, help='resnet18 / 3detr / mlp')
    parser.add_argument("--log_dir", type=str, default='output', help="log dir",)
    parser.add_argument("--use_mlp_in_attention", type=str2bool, default=False, help="use mlp to replace 3DeTR encoder & decoder")
    parser.add_argument("--arch_type", type=str, default='coarse_fine', help="architecture type: coarse_fine, coarse2fine, fine, coarse", )
    parser.add_argument("--fusion_type", type=str, default='self_attn', help="self_attn, mlp", )
    parser.add_argument("--preencoder_type", type=str, default='conv', help="resnet18, conv", )
    parser.add_argument("--focal_length", default=1.2, type=float, help='1.5, ')
    parser.add_argument('--nerf_render_rgb', type=str2bool, default=True, help="render rgb use nerf")
    parser.add_argument('--nerf_render_depth', type=str2bool, default=True, help="render depth use nerf")
    return parser


def do_train(
    args,
    model,
    model_no_ddp,
    optimizer,
    criterion,
    dataset_config,
    dataloaders,
    best_val_metrics,
):
    """
    Main training loop.
    This trains the model for `args.max_epoch` epochs and tests the model after every `args.eval_every_epoch`.
    We always evaluate the final checkpoint and report both the final AP and best AP on the val set.
    """

    num_iters_per_epoch = len(dataloaders["train"])
    num_iters_per_eval_epoch = len(dataloaders["test"])
    # args.max_epoch = int(args.max_epoch / dataset_config.pseudo_batch_size)
    args.max_epoch = int(args.max_epoch)
    print(f"Model is {model}")
    print(f"Training started at epoch {args.start_epoch} until {args.max_epoch}.")
    print(f"One training epoch = {num_iters_per_epoch} iters.")
    print(f"One eval epoch = {num_iters_per_eval_epoch} iters.")

    final_eval = os.path.join(args.checkpoint_dir, "final_eval.txt")
    final_eval_pkl = os.path.join(args.checkpoint_dir, "final_eval.pkl")

    if os.path.isfile(final_eval):
        print(f"Found final eval file {final_eval}. Skipping training.")
        return

    logger = Logger(args.checkpoint_dir)

    for epoch in range(args.start_epoch, args.max_epoch):
        if is_distributed():
            dataloaders["train_sampler"].set_epoch(epoch)

        aps = train_one_epoch(
            args,
            epoch,
            model,
            optimizer,
            criterion,
            dataset_config,
            dataloaders["train"],
            logger,
        )

        # latest checkpoint is always stored in checkpoint.pth
        save_checkpoint(
            args.checkpoint_dir,
            model_no_ddp,
            optimizer,
            epoch,
            args,
            best_val_metrics,
            filename="checkpoint.pth",
        )

        metrics = aps.compute_metrics()
        metric_str = aps.metrics_to_str(metrics, per_class=False)
        metrics_dict = aps.metrics_to_dict(metrics)
        curr_iter = epoch * len(dataloaders["train"])
        train_metrics_dict = metrics_dict
        if is_primary():
            print("==" * 10)
            print(f"Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
            print("==" * 10)
            logger.log_scalars(metrics_dict, curr_iter, prefix="Train/")
        if (
            epoch > 0
            and args.save_separate_checkpoint_every_epoch > 0
            and epoch % args.save_separate_checkpoint_every_epoch == 0
        ):
            # separate checkpoints are stored as checkpoint_{epoch}.pth
            save_checkpoint(
                args.checkpoint_dir,
                model_no_ddp,
                optimizer,
                epoch,
                args,
                best_val_metrics,
            )

        if False and (epoch % args.eval_every_epoch == 0 or epoch == (args.max_epoch - 1)):  # False and
            ap_calculator = evaluate(
                args,
                epoch,
                model,
                criterion,
                dataset_config,
                dataloaders["test"],
                logger,
                curr_iter,
            )
            metrics = ap_calculator.compute_metrics()
            ap25 = metrics[0.25]["mAP"]
            metric_str = ap_calculator.metrics_to_str(metrics, per_class=True)
            metrics_dict = ap_calculator.metrics_to_dict(metrics)
            test_metrics_dict = metrics_dict
            if is_primary():
                print("==" * 10)
                print(f"Evaluate Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
                print("==" * 10)
                logger.log_scalars(metrics_dict, curr_iter, prefix="Test/")

            if is_primary() and (
                len(best_val_metrics) == 0 or best_val_metrics[0.25]["mAP"] < ap25
            ):
                best_val_metrics = metrics
                filename = "checkpoint_best.pth"
                save_checkpoint(
                    args.checkpoint_dir,
                    model_no_ddp,
                    optimizer,
                    epoch,
                    args,
                    best_val_metrics,
                    filename=filename,
                )
                print(
                    f"Epoch [{epoch}/{args.max_epoch}] saved current best val checkpoint at {filename}; ap25 {ap25}"
                )
        else:
            test_metrics_dict = {}

        if args.USE_WANDB:
            log_stats = {**{f'train_{k}': v for k, v in train_metrics_dict.items()},
                         **{f'test_{k}': v for k, v in test_metrics_dict.items()}, 'epoch': epoch,}
            wandb.log(log_stats, step=epoch + 1)

    if False: # True:
        # always evaluate last checkpoint
        epoch = args.max_epoch - 1
        curr_iter = epoch * len(dataloaders["train"])
        ap_calculator = evaluate(
            args,
            epoch,
            model,
            criterion,
            dataset_config,
            dataloaders["test"],
            logger,
            curr_iter,
        )
        metrics = ap_calculator.compute_metrics()
        metric_str = ap_calculator.metrics_to_str(metrics)
        if is_primary():
            print("==" * 10)
            print(f"Evaluate Final [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
            print("==" * 10)

            with open(final_eval, "w") as fh:
                fh.write("Training Finished.\n")
                fh.write("==" * 10)
                fh.write("Final Eval Numbers.\n")
                fh.write(metric_str)
                fh.write("\n")
                fh.write("==" * 10)
                fh.write("Best Eval Numbers.\n")
                fh.write(ap_calculator.metrics_to_str(best_val_metrics))
                fh.write("\n")

            with open(final_eval_pkl, "wb") as fh:
                pickle.dump(metrics, fh)


def test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders):
    if args.test_ckpt is None or not os.path.isfile(args.test_ckpt):
        f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"
        sys.exit(1)

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"])
    logger = Logger()
    criterion = None  # do not compute loss for speed-up; Comment out to see test loss
    epoch = -1
    curr_iter = 0
    ap_calculator = evaluate(
        args,
        epoch,
        model,
        criterion,
        dataset_config,
        dataloaders["test"],
        logger,
        curr_iter,
    )
    metrics = ap_calculator.compute_metrics()
    metric_str = ap_calculator.metrics_to_str(metrics)
    if is_primary():
        print("==" * 10)
        print(f"Test model; Metrics {metric_str}")
        print("==" * 10)


def main(local_rank, args):
    if args.ngpus > 1:
        print(
            "Initializing Distributed Training. This is in BETA mode and hasn't been tested thoroughly. Use at your own risk :)"
        )
        print("To get the maximum speed-up consider reducing evaluations on val set by setting --eval_every_epoch to greater than 50")
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )

    print(f"Called with args: {args}")
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())

    datasets, dataset_config = build_dataset(args)
    model, _ = build_model(args, dataset_config)
    model = model.cuda(local_rank)
    model_no_ddp = model

    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
    criterion = build_criterion(args, dataset_config)
    criterion = criterion.cuda(local_rank)

    dataloaders = {}
    if args.test_only:
        dataset_splits = ["test"]
    else:
        dataset_splits = ["train", "test"]
    for split in dataset_splits:
        if split == "train":
            shuffle = False  # True
        else:
            shuffle = False
        if is_distributed():
            sampler = DistributedSampler(datasets[split], shuffle=shuffle)
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(datasets[split])
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=min(args.batchsize_per_gpu, args.dataset_num_workers),
            worker_init_fn=my_worker_init_fn,
        )
        dataloaders[split + "_sampler"] = sampler

    if args.test_only:
        criterion = None  # faster evaluation
        test_model(args, model, model_no_ddp, criterion, dataset_config, dataloaders)
    else:
        assert (
            args.checkpoint_dir is not None
        ), f"Please specify a checkpoint dir using --checkpoint_dir"
        if is_primary() and not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
        optimizer = build_optimizer(args, model_no_ddp)
        loaded_epoch, best_val_metrics = resume_if_possible(
            args.checkpoint_dir, model_no_ddp, optimizer
        )
        args.start_epoch = loaded_epoch + 1
        do_train(
            args,
            model,
            model_no_ddp,
            optimizer,
            criterion,
            dataset_config,
            dataloaders,
            best_val_metrics,
        )


def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    time_str = time.strftime("%Y%m%d-%H%M%S") + '-{}-{}-{}-{}-test_{}'.format(args.input_type, args.arch_type, args.fusion_type, args.model_type, args.test_only)
    args.log_dir = os.path.join(args.log_dir, time_str)
    os.makedirs(args.log_dir, exist_ok=True)
    args.checkpoint_dir = args.log_dir
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    if args.USE_WANDB:
        wandb.init(project='3detr', config=args, settings=wandb.Settings(_disable_stats=True))
        wandb.run.name = '{}_{}_{}'.format(wandb.run.name.split('-')[-1], args.dataset_name, time_str)
    launch_distributed(args)
