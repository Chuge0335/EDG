
import torch
import argparse
from tqdm import tqdm
from merge.model import load_model, load_camera_model
import inspect
import os
import re
from merge.utils import rgetattr
from lvdm.modules.attention import SpatialTransformer, TemporalTransformer
from lvdm.modules.networks.openaimodel3d import ResBlock, Downsample, Upsample

@torch.inference_mode()
def dyn(
    *,
    sft_path: str,
    dpo_path: str,
    save_dir: str,
    sft_config: str,
    dpo_config: str = 'configs/inference_512_v1.0.yaml',
    alpha: float,
):

    # global args
    # keys, _, _, values = inspect.getargvalues(inspect.currentframe())
    # args = argparse.Namespace(**{k:values[k] for k in keys})
    sft_model = load_model(sft_path, sft_config).to('cuda')
    dpo_model = load_model(dpo_path, dpo_config).to('cuda')
    # assert len(sft_model.state_dict()) == len(dpo_model.state_dict())
    total = len(dpo_model.state_dict())
    for name, dpo_model_param in tqdm(dpo_model.named_parameters(), total=total):
        sft_model_param = sft_model.state_dict()[name]
        sft_model_param.data = dpo_model_param.data + alpha * (dpo_model_param.data - sft_model_param.data)

    os.makedirs(save_dir, exist_ok=True)
    torch.save({'state_dict': sft_model.state_dict()}, save_dir + '/ckpt.pt')

@torch.inference_mode()
def dyn2(
    *,
    sft_path: str,
    dpo_path: str,
    save_dir: str,
    sft_config: str,
    dpo_config: str = 'configs/inference_512_v1.0.yaml',
    alpha: float,
):

    # global args
    # keys, _, _, values = inspect.getargvalues(inspect.currentframe())
    # args = argparse.Namespace(**{k:values[k] for k in keys})
    sft_model = load_model(sft_path, sft_config).to('cuda')
    dpo_model = load_model(dpo_path, dpo_config).to('cuda')
    # assert len(sft_model.state_dict()) == len(dpo_model.state_dict())
    total = len(dpo_model.state_dict())
    for name, dpo_model_param in tqdm(dpo_model.named_parameters(), total=total):
        sft_model_param = sft_model.state_dict()[name]
        dpo_model_param.data = dpo_model_param.data + alpha * (dpo_model_param.data - sft_model_param.data)

    os.makedirs(save_dir, exist_ok=True)
    torch.save({'state_dict': dpo_model.state_dict()}, save_dir + '/ckpt.pt')

@torch.inference_mode()
def main1(
    *,
    dynamic_path: str,
    consistency_path: str,
    sft_path: str,
    pre_path: str,
    base_path: str,
    save_dir: str,
    alpha: float,
):

    # global args
    # keys, _, _, values = inspect.getargvalues(inspect.currentframe())
    # args = argparse.Namespace(**{k:values[k] for k in keys})

    sft_models = [load_model(sft_path) for sft_path in [dynamic_path, consistency_path]]
    base_model = load_model(base_path)

    total = len(base_model.state_dict())

    for name, base_model_param in tqdm(base_model.named_parameters(), total=total):
        base_model_param.data = base_model_param.data + alpha * (
            sum([(sft_model.state_dict()[name].data - base_model_param.data) for sft_model in sft_models])
        )

    os.makedirs(save_dir, exist_ok=True)
    torch.save({'state_dict': base_model.state_dict()}, save_dir + '/ckpt.pt')

# TODO1: extrapolate different ability
# TODO2: migrate the interfernece  

@torch.inference_mode()
def main2(
    *,
    dynamic_path: str,
    consistency_path: str,
    sft_path: str,
    pre_path: str,
    base_path: str,
    save_dir: str,
    alpha: float,
    binary: str, # [spa, temporal, res, init_attn, out, image_proj] => [0,1,2,3,4,5]
):

    # global args
    # keys, _, _, values = inspect.getargvalues(inspect.currentframe())
    # args = argparse.Namespace(**{k:values[k] for k in keys})

    # sft_models = [load_model(sft_path) for sft_path in [dynamic_path, consistency_path]]
    dynamic_model = load_model(dynamic_path).to('cuda')
    consistency_model = load_model(pre_path).to('cuda')
    base_model = load_model(base_path).to('cuda')

    total = len(base_model.state_dict())

    for name, base_model_param in tqdm(base_model.named_parameters(), total=total):
        match = re.match(r'model\.diffusion_model\.(input|output)_blocks\.(\d+)\.(\d+)\.\w+', name)
        module = None
        if match:
            upper_name = '.'.join(name.split('.')[:5])
            module = rgetattr(base_model, upper_name)
        else:
            match2 = re.match(r'model\.diffusion_model\.middle_block\.(\d+)\.\w+', name)
            if match2:
                upper_name = '.'.join(name.split('.')[:4])
                module = rgetattr(base_model, upper_name)
        if binary[0] == '1' and isinstance(module, TemporalTransformer):
            base_model_param.data = base_model_param.data + alpha * (dynamic_model.state_dict()[name].data - base_model_param.data)
        elif binary[1] == '1' and isinstance(module, SpatialTransformer):
            base_model_param.data = base_model_param.data + alpha * (dynamic_model.state_dict()[name].data - base_model_param.data)
        elif binary[2] == '1' and isinstance(module, ResBlock):
            base_model_param.data = base_model_param.data + alpha * (dynamic_model.state_dict()[name].data - base_model_param.data)
        
        if binary[3] == '1' and 'diffusion_model.init_attn' in name:
            base_model_param.data = base_model_param.data + alpha * (dynamic_model.state_dict()[name].data - base_model_param.data)
        if binary[4] == '1' and 'diffusion_model.out' in name:
            base_model_param.data = base_model_param.data + alpha * (dynamic_model.state_dict()[name].data - base_model_param.data)
        if binary[5] == '1' and 'image_proj_model' in name:
            base_model_param.data = base_model_param.data + alpha * (dynamic_model.state_dict()[name].data - base_model_param.data)
        # if 'diffusion_model.fps_embedding' in name:
        #     base_model_param.data = base_model_param.data + alpha * (consistency_model.state_dict()[name].data - base_model_param.data)

    os.makedirs(save_dir, exist_ok=True)
    torch.save({'state_dict': base_model.state_dict()}, save_dir + '/ckpt.pt')


@torch.inference_mode()
def main3(
    *,
    dynamic_path: str,
    camera_path: str,
    camera_config: str,
    sft_path: str,
    pre_path: str,
    base_path: str,
    save_dir: str,
    alpha: float,
    binary: str, # [spa, temporal, res, init_attn, out, image_proj] => [0,1,2,3,4,5]
):

    # global args
    # keys, _, _, values = inspect.getargvalues(inspect.currentframe())
    # args = argparse.Namespace(**{k:values[k] for k in keys})

    # sft_models = [load_model(sft_path) for sft_path in [dynamic_path, consistency_path]]
    # dynamic_model = load_model(dynamic_path).to('cuda')
    import lvdm
    old = lvdm.modules.attention.TemporalTransformer
    lvdm.modules.attention.TemporalTransformer = TemporalTransformer2
    lvdm.modules.networks.openaimodel3d.TemporalTransformer = TemporalTransformer2
    
    camera_model = load_camera_model('/opt/tiger/DiT/ckpt/camera_motion_odata/epoch=6-step=14000.ckpt', temporal_selfatt_only=False, image_cross_attention=True).to('cuda')

    lvdm.modules.attention.TemporalTransformer=old
    lvdm.modules.networks.openaimodel3d.TemporalTransformer=old
    base_model = load_model(base_path).to('cuda')

    total = len(base_model.state_dict())

    for name, _ in tqdm(camera_model.named_parameters(), total=total):
        if name not in base_model.state_dict():
            continue
        camera_model.state_dict()[name].data = camera_model.state_dict()[name].data + alpha * (camera_model.state_dict()[name].data - base_model.state_dict()[name].data)
        # match = re.match(r'model\.diffusion_model\.(input|output)_blocks\.(\d+)\.(\d+)\.\w+', name)
        # module = None
        # if match:
        #     upper_name = '.'.join(name.split('.')[:5])
        #     module = rgetattr(base_model, upper_name)
        # else:
        #     match2 = re.match(r'model\.diffusion_model\.middle_block\.(\d+)\.\w+', name)
        #     if match2:
        #         upper_name = '.'.join(name.split('.')[:4])
        #         module = rgetattr(base_model, upper_name)
        # if binary[0] == '1' and isinstance(module, TemporalTransformer):
        #     base_model_param.data = base_model_param.data + alpha * (camera_path.state_dict()[name].data - base_model_param.data)
        # elif binary[1] == '1' and isinstance(module, SpatialTransformer):
        #     base_model_param.data = base_model_param.data + alpha * (camera_path.state_dict()[name].data - base_model_param.data)
        # elif binary[2] == '1' and isinstance(module, ResBlock):
        #     base_model_param.data = base_model_param.data + alpha * (camera_path.state_dict()[name].data - base_model_param.data)
        
        # if binary[3] == '1' and 'diffusion_model.init_attn' in name:
        #     base_model_param.data = base_model_param.data + alpha * (camera_path.state_dict()[name].data - base_model_param.data)
        # if binary[4] == '1' and 'diffusion_model.out' in name:
        #     base_model_param.data = base_model_param.data + alpha * (camera_path.state_dict()[name].data - base_model_param.data)
        # if binary[5] == '1' and 'image_proj_model' in name:
        #     base_model_param.data = base_model_param.data + alpha * (camera_path.state_dict()[name].data - base_model_param.data)
        # if 'diffusion_model.fps_embedding' in name:
        #     base_model_param.data = base_model_param.data + alpha * (consistency_model.state_dict()[name].data - base_model_param.data)

    os.makedirs(save_dir, exist_ok=True)
    torch.save({'state_dict': camera_model.state_dict()}, save_dir + '/ckpt.pt')


if __name__ == '__main__':
    import defopt
    try:
        defopt.run(eval(os.getenv('func', 'main')))
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)
    