# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from collections import defaultdict
import json


DICT_LRD = {"retfound":"hf",
            "swinv2":"hf",
            "dinov2":"hf",
            "visionfm":"hf",
            "mflm":"mae",
            "mae_st":"mae",
            "video_mae":"hf",
            "video_mae2": "mae",
            "omni_mae": "omni_mae",
            "vjepa": "hf"
            }

def param_groups_lrd(model, model_name, weight_decay=0.05, layer_decay=.75):
    if "ResNet" in model._get_name():
        return model.parameters()

    if DICT_LRD.get(model_name) == "mae":
        return get_param_groups_mae(model, weight_decay=weight_decay, no_weight_decay_list=model.no_weight_decay(),
                                    layer_decay=layer_decay)
    
    elif DICT_LRD.get(model_name) == "hf":
        param_groups = get_param_groups_hf(model, weight_decay=weight_decay, lr_decay_rate=layer_decay)
        # print(param_groups)
        return param_groups
    
    elif DICT_LRD.get(model_name) == "omni_mae":
        return get_param_groups_omnimae(model, weight_decay=weight_decay, no_weight_decay_list=model.trunk.no_weight_decay(),
                                    layer_decay=layer_decay)
    
    else:
        raise NotImplementedError("The model chosen doesn't have a layer-wise decay strategy.")


def get_param_groups_hf(model, weight_decay=0.05, lr_decay_rate=1.0, patch_embed_lr_mult=1.0, classifier_lr_mult=1):
    # From https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/utils/param_groups.py#L52
    chunked_blocks = False
    if hasattr(model, "n_blocks"):
        n_blocks = model.n_blocks
        chunked_blocks = model.chunked_blocks
    elif hasattr(model, "blocks"):
        n_blocks = len(model.blocks)
    elif hasattr(model, "backbone"):
        n_blocks = len(model.backbone.blocks)
    elif hasattr(model, "config"):
        n_blocks = model.config.num_hidden_layers
    else:
        n_blocks = 0
    all_param_groups = []

    for name, param in model.named_parameters():
        name = name.replace("_fsdp_wrapped_module.", "")
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(
            name, lr_decay_rate, num_layers=n_blocks, force_is_backbone=n_blocks > 0, chunked_blocks=chunked_blocks
        )
        d = {"params": param, "is_last_layer": False, "lr_scale": decay_rate, "weight_decay":weight_decay, "wd_multiplier": 1.0, "name": name}

        if "last_layer" in name:
            d.update({"is_last_layer": True})

        if name.endswith(".bias") or "norm" in name or "gamma" in name or "lambda" in name:
            d.update({"weigth_decay": 0.0})

        if "patch_embed" in name or "patch_embeddings" in name:
            d.update({"lr_scale": d["lr_scale"] * patch_embed_lr_mult})
        
        if "classifier" in name:
            d.update({"lr_scale": d["lr_scale"] * classifier_lr_mult})

        all_param_groups.append(d)

    return all_param_groups


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12, force_is_backbone=False, chunked_blocks=False):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone") or force_is_backbone:
        if (
                ".pos_embed" in name
                or ".patch_embed" in name
                or ".mask_token" in name
                or ".cls_token" in name
                or ".register_tokens" in name
        ):
            layer_id = 0
        elif force_is_backbone and (
                "position_embeddings" in name
                or "patch_embeddings" in name
                or "mask_token" in name
                or "cls_token" in name
                or "register_tokens" in name
        ):
            layer_id = 0
        elif ".encoder.layer." in name and "residual" not in name:
            layer_id = int(name[name.find(".encoder."):].split(".")[3]) + 1
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks."):].split(".")[2]) + 1
        elif chunked_blocks and "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks."):].split(".")[2]) + 1
        elif "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks."):].split(".")[1]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_param_groups_mae(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'mask_token']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith("pos_embed"):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (not bias_wd)
            and len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]

def get_param_groups_omnimae(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = model.trunk.get_num_layers() 

    layer_scales = [
        layer_decay ** (num_layers - i) for i in range(num_layers + 1)
    ]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = model.trunk.get_layer_id(n)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())