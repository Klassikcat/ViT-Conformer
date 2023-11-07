from torch import Tensor, nn


class BeamSearchCTC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features: Tensor) -> Tensor:
        pass
