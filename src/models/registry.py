from functools import partial

import torch
import torch.nn as nn
from transformers import Dinov2ForImageClassification

from src.models import HFVIT, VisionFM, Vjepa
from src.models.models_mgmt import freeze_model, load_lora_model, load_pretrained_vjepa_model




def retfound(**kwargs):
    print("Hugging Face model loaded")

    model = HFVIT.from_pretrained(kwargs.get("finetune_vit"))

    if kwargs.get("lora"):
        model = load_lora_model(model)

    if kwargs.get("freeze_backbone"):
        freeze_model(model)

    return model


def dinov2(**kwargs):
    model = Dinov2ForImageClassification.from_pretrained(
            kwargs.get("finetune_vit"), num_labels=kwargs["num_classes"]
        )

    if kwargs.get("lora"):
        model = load_lora_model(model)

    if kwargs.get("freeze_backbone"):
        freeze_model(model)

    return model


def visionfm(**kwargs):
    return VisionFM(kwargs.get("finetune_vit"), **kwargs)


def vjepa(**kwargs):
    model = Vjepa(num_classes=kwargs.get("num_classes"), freeze_backbone=kwargs.get("freeze_backbone"))
    load_pretrained_vjepa_model(model, kwargs.get("finetune_vit"), "encoder")
    return model
