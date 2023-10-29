import torch


class VitConformerHandler:
    def __init__(self):
        pass

    def model_fn(self, model_dir: str):
        pass

    def input_fn(self, input_data: dict, content_type: str):
        pass

    def predict_fn(self, data: torch.Tensor, model):
        pass

    def output_fn(self, prediction: torch.Tensor, accept):
        pass
