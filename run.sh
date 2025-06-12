#!/bin/bash

# Step 1: Extrapolating and Merging
CKPT_PATH="$PWD/ckpt"
noise="$CKPT_PATH/dynamicrafter_analytic_init/initial_noise_512.pt"

python3 run_merge.py \
--basedir $PWD \
--ckptdir $CKPT_PATH \
--outdir $CKPT_PATH/merge/ \
--dyn-alpha 0.25 \
--dyn-beta 0.75 \
--con-alpha 1.0 \
--con-beta 0.25 \


# Step 2: Decoupled Injection and Sampling
dynamic_path="$CKPT_PATH/merge/dyn.pt" 
consistency_path="$CKPT_PATH/merge/con.pt" 

CUDA_VISIBLE_DEVICES=0 \
torchrun --master_port 23401 --nproc_per_node=1 \
evaluation/ddp_wrapper.py \
--module infer_multi \
--seed 123 \
--ckpt_path $ckpt \
--config configs/inference_512_v1.0.yaml \
--savedir ./output \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir ./prompts \
--text_input \
--video_length 16 \
--frame_stride 24 \
--timestep_spacing 'uniform_trailing' \
--guidance_rescale 0.7 \
--perframe_ae \
--M 1000 \
--whether_analytic_init 0 \
--analytic_init_path $noise \
--dynamic_path $dynamic_path \
--consistency_path $consistency_path \
--T 500 