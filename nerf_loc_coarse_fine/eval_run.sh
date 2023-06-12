# 1。 Fine, 3DeTR, nerf
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type fine --input_type nerf --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220829-222621-nerf-fine-self_attn-3detr-test_False/checkpoint.pth
# 2. Fine, 3DeTR, rendered_rgb
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type fine --input_type rendered_rgb --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220830-111843-rendered_rgb-fine-self_attn-3detr-test_False/checkpoint.pth
# 3. Fine, 3DeTR, rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type fine --input_type rendered_depth --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220830-170131-rendered_depth-fine-self_attn-3detr-test_False/checkpoint.pth
# 4. Fine, 3DeTR, nerf_rendered_rgb
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type fine --input_type nerf_rendered_rgb --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220830-230327-nerf_rendered_rgb-fine-self_attn-3detr-test_False/checkpoint.pth
# 5. Fine, 3DeTR, nerf_rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type fine --input_type nerf_rendered_depth --model_type 3detr --USE_WANDB False --test_only --test_ckpt
# 6. Fine, 3DeTR, rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type fine --input_type rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB False --test_only --test_ckpt
# 7. Fine, 3DeTR, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type fine --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220910-142155-nerf_rendered_rgb_rendered_depth-fine-self_attn-3detr-test_False/checkpoint.pth

# Coarse, 3DeTR, nerf
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse --input_type nerf --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220905-213401-nerf-coarse-self_attn-3detr-test_False/checkpoint.pth
# Coarse, 3DeTR, rendered_rgb
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse --input_type rendered_rgb --model_type 3detr --USE_WANDB False --test_only --test_ckpt
# Coarse, 3DeTR, rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse --input_type rendered_depth --model_type 3detr --USE_WANDB False --test_only --test_ckpt
# Coarse, 3DeTR, nerf_rendered_rgb
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse --input_type nerf_rendered_rgb --model_type 3detr --USE_WANDB False --test_only --test_ckpt
# Coarse, 3DeTR, nerf_rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse --input_type nerf_rendered_depth --model_type 3detr --USE_WANDB False --test_only --test_ckpt
# Coarse, 3DeTR, rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse --input_type rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB False --test_only --test_ckpt
# Coarse, 3DeTR, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB False --test_only --test_ckpt

# Coarse-Fine, 3DeTR, nerf
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220901-130439-nerf-coarse_fine-self_attn-3detr-test_False/checkpoint.pth
# Coarse-Fine, 3DeTR, rendered_rgb
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_rgb --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220902-114107-rendered_rgb-coarse_fine-self_attn-3detr-test_False/checkpoint.pth
# Coarse-Fine, 3DeTR, rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_depth --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220902-195243-rendered_depth-coarse_fine-self_attn-3detr-test_False/checkpoint.pth
# Coarse-Fine, 3DeTR, nerf_rendered_rgb
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_rgb --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220903-210449-nerf_rendered_rgb-coarse_fine-self_attn-3detr-test_False/checkpoint.pth
# Coarse-Fine, 3DeTR, nerf_rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_depth --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220904-143939-nerf_rendered_depth-coarse_fine-self_attn-3detr-test_False/checkpoint.pth
# Coarse-Fine, 3DeTR, rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse_fine --input_type rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220904-230412-rendered_rgb_rendered_depth-coarse_fine-self_attn-3detr-test_False/checkpoint.pth
# Coarse-Fine, 3DeTR, nerf_rendered_rgb_rendered_depth
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf_rendered_rgb_rendered_depth --model_type 3detr --USE_WANDB False --test_only --test_ckpt output/20220904-230732-nerf_rendered_rgb_rendered_depth-coarse_fine-self_attn-3detr-test_False/checkpoint.pth

# Coarse-Fine, 3DeTR, nerf, mlp
CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name nerf --arch_type coarse_fine --input_type nerf --model_type 3detr --fusion_type mlp --USE_WANDB False --test_only --test_ckpt output/20220905-122735-nerf-coarse_fine-mlp-3detr-test_False/checkpoint.pth
