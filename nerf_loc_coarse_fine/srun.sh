# Train
srun --gres=gpu:GTX1080Ti:1 --nodelist=chpc-gpu002 python -u main.py --dataset_name nerf --checkpoint_dir output
# Test
# python main.py --dataset_name <dataset_name> --nqueries <number of queries> --test_ckpt <path_to_checkpoint> --test_only [--enc_type masked]