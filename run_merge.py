from email.mime import base
import torch
from collections import defaultdict, OrderedDict
import tqdm
import re
import torch.nn as nn
import copy
import json
import sys
import transformers
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
import os
import functools
from peft import LoraConfig,get_peft_model
from collections import defaultdict, OrderedDict
import torch.nn.functional as F 
import torch
from collections import defaultdict
import numpy as np
import inspect
import pandas as pd
from merge.model import load_model
import merge.sparsify as sparsify
import merge.utils as utils
from merge.param import param


@torch.inference_mode()
def cil_merge(
    *,
    basedir: str,
    ckptdir: str,
    outdir: str,
    dyn_alpha: float = 0.,
    dyn_beta: float = 0.,
    con_alpha: float = 0.,
    con_beta: float = 0.,
):
    os.makedirs(outdir, exist_ok=True)

    sft_model = load_model(
        os.path.join(ckptdir, './dynamicrafter_cil_512_v1/timenoise.ckpt'),
        os.path.join(basedir, './configs/inference_512_v1.0.yaml')
    )
    pre_model = load_model(
        os.path.join(ckptdir, './dynamicrafter_512_v1/model.ckpt'),
        os.path.join(basedir, './configs/inference_512_v1.0.yaml')
    )

    _model = copy.deepcopy(sft_model)

    sft_model = param(sft_model)
    pre_model = param(pre_model)
    dyn_model = pre_model + dyn_alpha * (pre_model - sft_model)

    def match_new_para(n):
        return bool(re.fullmatch(r"model\.diffusion_model\.(input_blocks|output_blocks|middle_block|init_attn)\..*\.(queries|qformer\..*|cc_projection\..*)", n))

    def get_adt(mask_rate=0.7):
        mask_rate = float(mask_rate)
        masked_param = sft_model - pre_model
        for n, p in masked_param.param_dict.items():
            if not torch.all(p == 0).item():
                print('non zero ', n)
        masked_param = masked_param.map(
            lambda n,p: sparsify.bernoulli(
                p, 
                1 - mask_rate,
            ),
            # ) if (not match_new_para(n) or not torch.all(p == 0).item()) else p,
            desc='bernoulli'
        )
        # keep left
        return masked_param

    def get_deg(mask_rate=0.7):
        mask_rate = float(mask_rate)
        masked_param = dyn_model - pre_model
        for n, p in masked_param.param_dict.items():
            if not torch.all(p == 0).item():
                print('non zero ', n)
        masked_param = masked_param.map(
            lambda n,p: sparsify.bernoulli(
                p, 
                1 - mask_rate,
            ) if not torch.all(p == 0).item() else p,
            desc='bernoulli'
        )
        return masked_param

    adt_model = get_adt()
    deg_model = get_deg()
    con_model = sft_model - adt_model

    # dyn*
    # task_vectors = [
    #     model - pre_model
    #     for model in [deg_model, adt_model]
    # ]
    # order matters! keep left model 
    merged_param = param(copy.deepcopy(adt_model))
    for n, p in merged_param.param_dict.items():
        if n in pre_model:
            if not torch.all(p == 0).item():
                print('merge', n)
            merged_param[n] = dyn_beta * deg_model[n] + pre_model[n]
    torch.save({'state_dict': merged_param.param_dict}, outdir + '/dyn.pt')

    merged_param = param(copy.deepcopy(adt_model))
    for n, p in merged_param.param_dict.items():
        if n in pre_model:
            if not torch.all(p == 0).item():
                print('merge', n)
            # + deg_model[n]
            merged_param[n] =  con_alpha * sft_model[n] + con_beta * adt_model[n] 
    torch.save({'state_dict': merged_param.param_dict}, outdir + '/con.pt')

if __name__ == '__main__':
    try:
        import defopt
        defopt.run(cil_merge)  
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)
