# 1。 Fine, 3DeTR, nerf
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type nerf --model_type 3detr --USE_WANDB True
# 2. Fine, 3DeTR, rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type rendered_rgb --model_type 3detr --USE_WANDB True
# 3. Fine, 3DeTR, rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type rendered_depth --model_type 3detr --USE_WANDB True
# 4. Fine, 3DeTR, nerf_rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type nerf_rendered_rgb --model_type 3detr --USE_WANDB True
# 5. Fine, 3DeTR, nerf_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type nerf_rendered_depth --model_type 3detr --USE_WANDB True
# 6. Fine, 3DeTR, rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True
# 7. Fine, 3DeTR, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True

# 8. Coarse, 3DeTR, nerf
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type nerf --model_type 3detr --USE_WANDB True
# 9. Coarse, 3DeTR, rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type rendered_rgb --model_type 3detr --USE_WANDB True
# 10. Coarse, 3DeTR, rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type rendered_depth --model_type 3detr --USE_WANDB True
# 11. Coarse, 3DeTR, nerf_rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type nerf_rendered_rgb --model_type 3detr --USE_WANDB True
# 12. Coarse, 3DeTR, nerf_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type nerf_rendered_depth --model_type 3detr --USE_WANDB True
# 13. Coarse, 3DeTR, rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True
# 14. Coarse, 3DeTR, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True

# 15. Coarse-Fine, 3DeTR, nerf, self_attn
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf --model_type 3detr --USE_WANDB True
# 16. Coarse-Fine, 3DeTR, rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_rgb --model_type 3detr --USE_WANDB True
# 17. Coarse-Fine, 3DeTR, rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_depth --model_type 3detr --USE_WANDB True
# 18. Coarse-Fine, 3DeTR, nerf_rendered_rgb
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_rgb --model_type 3detr --USE_WANDB True
# 19. Coarse-Fine, 3DeTR, nerf_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_depth --model_type 3detr --USE_WANDB True
# 20. Coarse-Fine, 3DeTR, rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True
# 21. Coarse-Fine, 3DeTR, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True

# Coarse-Fine, 3DeTR, nerf, mlp
CUDA_VISIBLE_DEVICES=4 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf --model_type 3detr --USE_WANDB True --fusion_type mlp

# ScanNet
# 22. Coarse-Fine, 3DeTR, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_root_dir ../scannet_dataset/ --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB True
