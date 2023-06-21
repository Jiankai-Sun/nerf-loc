# NeRF-Loc



## Installation
### Installation for mmdetection3d
```
pip install nuscenes-devkit
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -e .
```
### Installation for NeRF-Loc
```
cd nerf_loc_coarse_fine
conda create -f environment.yml
conda activate nerf
```

## Dataset
### Objectron for NeRF
`Objectron/notebooks/pytorch_Objectron_NeRF_Tutorial.ipynb`

Download data
```
cd Objectron/notebooks
python Parse_Annotations.py
```

## How to Use
### Train NeRf
Install the conda environment and train NeRF based on `nerf-pytorch` instructions.
```
cd packages/nerf-pytorch
python run_multi_nerf.py --config configs/objectron/objectron_1.txt --no_ndc
```

### Train NeRF-Loc
```
NERF_LOC_WORK_DIR=nerf_loc_coarse_fine
cd $NERF_LOC_WORK_DIR
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf
# Coarse_Fine, NeRF-Loc, rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_depth --model_type 3detr --USE_WANDB True

# Fine, NeRF-Loc, nerf
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type fine --input_type nerf --model_type 3detr --USE_WANDB True
# Fine, NeRF-Loc, rendered_rgb
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type fine --input_type rendered_rgb --model_type 3detr --USE_WANDB True
# Fine, NeRF-Loc, rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type fine --input_type rendered_depth --model_type 3detr --USE_WANDB True
# Fine, NeRF-Loc, nerf_rendered_rgb
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type fine --input_type nerf_rendered_rgb --model_type 3detr --USE_WANDB True
# Fine, NeRF-Loc, nerf_rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type fine --input_type nerf_rendered_depth --model_type 3detr --USE_WANDB True
# Fine, NeRF-Loc, rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type fine --input_type rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True
# Fine, NeRF-Loc, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type fine --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True

# Coarse, NeRF-Loc, nerf
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse --input_type nerf --model_type 3detr --USE_WANDB True
# Coarse, NeRF-Loc, rendered_rgb
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse --input_type rendered_rgb --model_type 3detr --USE_WANDB True
# Coarse, NeRF-Loc, rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse --input_type rendered_depth --model_type 3detr --USE_WANDB True
# Coarse, NeRF-Loc, nerf_rendered_rgb
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse --input_type nerf_rendered_rgb --model_type 3detr --USE_WANDB True
# Coarse, NeRF-Loc, nerf_rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse --input_type nerf_rendered_depth --model_type 3detr --USE_WANDB True
# Coarse, NeRF-Loc, rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse --input_type rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True
# Coarse, NeRF-Loc, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True

# Coarse-Fine, NeRF-Loc, nerf
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf --model_type 3detr --USE_WANDB True
# Coarse-Fine, NeRF-Loc, rendered_rgb
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_rgb --model_type 3detr --USE_WANDB True
# Coarse-Fine, NeRF-Loc, rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_depth --model_type 3detr --USE_WANDB True
# Coarse-Fine, NeRF-Loc, nerf_rendered_rgb
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_rgb --model_type 3detr --USE_WANDB True
# Coarse-Fine, NeRF-Loc, nerf_rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_depth --model_type 3detr --USE_WANDB True
# Coarse-Fine, NeRF-Loc, rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True
# Coarse-Fine, NeRF-Loc, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True

# Fine, ResNet18, rendered_rgb
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type fine --input_type rendered_rgb --model_type resnet18 --USE_WANDB True
```
### Eval NeRF-Loc
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name nerf --arch_type fine --test_ckpt /mnt/ssd/jack/Programs/nerf-detection/packages/nerf_loc_coarse_fine/output/20220803-230552/checkpoint.pth --test_only --USE_WANDB False
## visualization
cd ../Objectron/notebooks
python vis_pred.py
```

## Acknowledgement
Our code is based on [3detr](https://github.com/facebookresearch/3detr) and [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch).