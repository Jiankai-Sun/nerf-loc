# nerf-detection

## Installation for mmdetection3d
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
## Objectron for NeRF
packages/Objectron/notebooks/pytorch_Objectron_NeRF_Tutorial.ipynb

Download data
```
cd packages/Objectron/notebooks
python Parse_Annotations.py
```

## ScanNet for NeRF
Download data for `scene0113_00` and put the data to `/home/jsun/data/Programs/nerf-detection/packages/mmdetection3d/data/scannet/scans/scene0113_00`. Follow here `https://github.com/open-mmlab/mmdetection3d/tree/30ad1aae13fe78e5b91d6d6f9eee835c1c086612/data/scannet`
```
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d/data/scannet
python batch_load_scannet_data.py
python extract_posed_images.py
cd ../../../tools
python scannet2ptNeRF.py
# Vis
cd ../Objectron/notebooks
python vis_pred_scannet.py
```

## Train NeRf
```
cd packages/nerf-pytorch
python run_multi_nerf.py --config configs/objectron/objectron_1.txt --no_ndc
```
Train NeRF on ScanNet
```
cd packages/nerf-pytorch
python run_multi_nerf_scannet.py --config configs/objectron/objectron_1.txt --no_ndc
``` 
## Train NeRF-Loc
```
NERF_LOC_WORK_DIR=nerf_loc_coarse_fine
cd $NERF_LOC_WORK_DIR
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf
# Coarse_Fine, NeRF-Loc, rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_depth --model_type 3detr --USE_WANDB True

# Fine, NeRF-Loc, nerf
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type nerf --model_type 3detr --USE_WANDB True
# Fine, NeRF-Loc, rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type rendered_rgb --model_type 3detr --USE_WANDB True
# Fine, NeRF-Loc, rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type rendered_depth --model_type 3detr --USE_WANDB True
# Fine, NeRF-Loc, nerf_rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type nerf_rendered_rgb --model_type 3detr --USE_WANDB True
# Fine, NeRF-Loc, nerf_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type nerf_rendered_depth --model_type 3detr --USE_WANDB True
# Fine, NeRF-Loc, rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True
# Fine, NeRF-Loc, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True

# Coarse, NeRF-Loc, nerf
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type nerf --model_type 3detr --USE_WANDB True
# Coarse, NeRF-Loc, rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type rendered_rgb --model_type 3detr --USE_WANDB True
# Coarse, NeRF-Loc, rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type rendered_depth --model_type 3detr --USE_WANDB True
# Coarse, NeRF-Loc, nerf_rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type nerf_rendered_rgb --model_type 3detr --USE_WANDB True
# Coarse, NeRF-Loc, nerf_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type nerf_rendered_depth --model_type 3detr --USE_WANDB True
# Coarse, NeRF-Loc, rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True
# Coarse, NeRF-Loc, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True

# Coarse-Fine, NeRF-Loc, nerf
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf --model_type 3detr --USE_WANDB True
# Coarse-Fine, NeRF-Loc, rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_rgb --model_type 3detr --USE_WANDB True
# Coarse-Fine, NeRF-Loc, rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_depth --model_type 3detr --USE_WANDB True
# Coarse-Fine, NeRF-Loc, nerf_rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_rgb --model_type 3detr --USE_WANDB True
# Coarse-Fine, NeRF-Loc, nerf_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_depth --model_type 3detr --USE_WANDB True
# Coarse-Fine, NeRF-Loc, rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True
# Coarse-Fine, NeRF-Loc, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True

# Fine, ResNet18, rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type rendered_rgb --model_type resnet18 --USE_WANDB True
```
## Eval NeRF-Loc
```
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --test_ckpt /mnt/ssd/jack/Programs/nerf-detection/packages/nerf_loc_coarse_fine/output/20220803-230552/checkpoint.pth --test_only --USE_WANDB False
## visualization
cd ../Objectron/notebooks
python vis_pred.py
```

## How to train PoseCNN

```
bash experiments/scripts/objectron_train.sh
```