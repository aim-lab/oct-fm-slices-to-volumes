from transformers import ViTForImageClassification


class HFVIT(ViTForImageClassification):
    def forward(self, *args, **kwargs):
        # Call the parent class's forward method with the updated interpolate value
        return super().forward(*args, interpolate_pos_encoding=True, **kwargs)
