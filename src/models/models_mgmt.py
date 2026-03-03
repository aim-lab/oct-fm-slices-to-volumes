import os

import torch
from peft import LoraConfig, get_peft_model

from src.utils.dist_utils import is_main_process
from src.utils.misc import print_trainable_parameters


def freeze_model(model):
    if hasattr(model,"base_model"):
        for param in model.base_model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = False


def load_lora_model(model):
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )

    lora_model = get_peft_model(model, config)
    print_trainable_parameters(lora_model)

    return lora_model


def save_model(args, epoch, fold, path_save, model, model_without_ddp, optimizer, loss_scaler):

    if loss_scaler is not None:
        checkpoint_paths = [path_save + '/checkpoint-best_fold_{}.pth'.format(fold)]
        # checkpoint_paths = [path_save + '/checkpoint-best.pth']
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.task, tag="checkpoint-best", client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    # TODO: manage lora loading
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def convert_slowfast_state_dict(state_dict):
    """
    Convert the state dict structure from SlowFast framework to the original Mae_ST.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k

        # Handle the special case for decoder_norm
        if k.startswith("pred_head.transforms.0.4."):
            new_k = "decoder_norm." + ".".join(k.split(".")[4:])  # Remove unnecessary index

        # Handle the special case for decoder_pred
        elif k.startswith("pred_head.projections.0."):
            new_k = "decoder_pred." + ".".join(k.split(".")[3:])  # Adjust projection layers

        # General case: Map pred_head.transforms.0.X -> decoder_blocks.X
        elif k.startswith("pred_head.transforms.0."):
            parts = k.split(".")
            block_id = int(parts[3])  # Extract the block index (0,1,2,3)
            new_k = f"decoder_blocks.{block_id}." + ".".join(parts[4:])
        
        # Assign to new state dict
        new_state_dict[new_k] = v

    return new_state_dict


def load_pretrained_vjepa_model(vjepa_model, pretrained, checkpoint_key='target_encoder'):
    if os.path.isfile(pretrained):
        if "probe" in pretrained:
            load_pretrained_vjepa_finetuned(vjepa_model, pretrained, "classifier", checkpoint_key)
        else:
            load_pretrained_vjepa_encoder(vjepa_model.encoder, pretrained, checkpoint_key)


def load_pretrained_vjepa_encoder(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    print(f'Loading pretrained encoder from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            print(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            print(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    print(f'loaded pretrained model with msg: {msg}')
    print(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint


def load_pretrained_vjepa_finetuned(
    vjepa_model,
    pretrained,
    classifier_key,
    checkpoint_key='target_encoder'
):
    print(f'Loading pretrained model from {pretrained}')

    load_pretrained_vjepa_encoder(vjepa_model.encoder, "../vjepa_vitl16.pth.tar", checkpoint_key)

    checkpoint = torch.load(pretrained, map_location='cpu')
    pretrained_dict = checkpoint[classifier_key]

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    for k, v in vjepa_model.classifier.state_dict().items():
        if k not in pretrained_dict:
            print(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            print(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = vjepa_model.classifier.load_state_dict(pretrained_dict, strict=False)
    print(vjepa_model.classifier)
    print(f'loaded pretrained model with msg: {msg}')
    print(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint