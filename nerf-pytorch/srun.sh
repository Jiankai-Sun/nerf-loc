srun --gres=gpu:GTX1080Ti:1 --nodelist=chpc-gpu002 python run_nerf.py --config configs/objectron_small.txt --render_only --get_nerf_repr
