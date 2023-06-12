# Train
CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --input_type rendered_rgb --model_type resnet18
# Test
# python main.py --dataset_name <dataset_name> --nqueries <number of queries> --test_ckpt <path_to_checkpoint> --test_only [--enc_type masked]
#CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --test_ckpt /mnt/ssd/jack/Programs/nerf-detection/packages/3detr_v3_coarse_fine/output/20220724-204127/checkpoint.pth --test_only --USE_WANDB False
#CUDA_VISIBLE_DEVICES=2 python main.py --dataset_name nerf --arch_type fine --test_ckpt /mnt/ssd/jack/Programs/nerf-detection/packages/3detr_v3_coarse_fine/output/20220803-230552/checkpoint.pth --test_only --USE_WANDB False