# <img src="assets/logo.png" alt="Logo" width="20" height="20"> EDG (**E**xtrapolating and **D**ecoupling **G**eneration)

<p align="center">
    <img src="assets/edg.png" width="400"/>
</p>

## 📰 News

🎉 Our paper *"Extrapolating and Decoupling Image-to-Video Generation Models: Motion Modeling is Easier Than You Think"* has been accepted as a Highlight at *CVPR 2025*! Check it out on [arXiv](https://arxiv.org/abs/2503.00948).


## ⚙️ Setup

### 🔧 Recommended: Anaconda

```bash
conda create -n dynamicrafter python=3.10 -y
conda activate dynamicrafter
pip install -r requirements.txt
sudo apt-get install libgl1-mesa-glx unzip -y
```


## 📥 Model Checkpoints

Download the required models from Hugging Face:

```bash
huggingface-cli download Doubiiu/DynamiCrafter_512 --local-dir ckpt/dynamicrafter_512_v1/
huggingface-cli download GraceZhao/DynamiCrafter-CIL-512 --local-dir ckpt/dynamicrafter_cil_512_v1/
huggingface-cli download GraceZhao/DynamiCrafter-Analytic-Init --local-dir ckpt/dynamicrafter_analytic_init/
```


## 💫 Inference

To run extrapolation, decoupled sampling, and generation, simply execute the `run.sh` script.
Ensure all required checkpoints are placed under the `CKPT_PATH` directory beforehand.

### 🔄 Model Merging & Extrapolation

We merge a CLI fine-tuned model with the original pretrained model (`dynamicrafter_512_v1`).
Motion dynamics are controlled by `dyn-alpha` and `dyn-beta`, while video consistency is controlled by `con-alpha` and `con-beta`:

```bash
python3 run_merge.py \
--basedir $PWD \
--ckptdir $CKPT_PATH \
--outdir $CKPT_PATH/merge/ \
--dyn-alpha 0.25 \
--dyn-beta 0.75 \
--con-alpha 1.0 \
--con-beta 0.25
```

### 🧩 Decoupled Sampling

Results will be saved in `output/<M>`, where `<M>` is the starting timestep for DDPM denoising.
By default, `M=1000` disables [CIL](https://github.com/thu-ml/cond-image-leakage.git).

```bash
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
```


## 📖 Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{tian2025extrapolatingdecouplingimagetovideogeneration,
  title={Extrapolating and Decoupling Image-to-Video Generation Models: Motion Modeling is Easier Than You Think}, 
  author={Jie Tian and Xiaoye Qu and Zhenyi Lu and Wei Wei and Sichen Liu and Yu Cheng},
  year={2025},
  eprint={2503.00948},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.00948}, 
}
```

