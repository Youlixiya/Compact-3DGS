#garden
CUDA_VISIBLE_DEVICES=1 python train.py --source_path data/360_v2/garden --images images_4 --model_path output/360_v2/triplane_garden

CUDA_VISIBLE_DEVICES=0 python train.py --source_path data/nerf_llff_data/fern --images images_4 --model_path output/llff/fern