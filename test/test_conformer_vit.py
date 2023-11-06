import torch
try:
    from vit_conformer.lightning import ViTConformerModel, ConformerModel
except ModuleNotFoundError:
    from src.vit_conformer.lightning import ViTConformerModel, ConformerModel

if __name__ == '__main__':
    model = ViTConformerModel(num_vocab=10)
    base_model = ConformerModel(num_vocab=10)

    tensor = torch.randn(1, 1, 80, 100)
    model(tensor)
    base_model(tensor)