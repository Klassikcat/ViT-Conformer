from typing import *

from lightning.pytorch.utilities.types import STEP_OUTPUT, EVAL_DATALOADERS
from omegaconf import DictConfig
from pathlib import Path, PosixPath

import torch
from torch import nn, Tensor
import lightning.pytorch as ptl
from ..nn.conformer import ConformerBlock, VitConformerBlock


class ConformerModel(ptl.LightningModule):
    def __init__(
            self,
    ) -> None:
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        pass

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        pass


class ViTConformerModel(ptl.LightningModule):
    def __init__(
            self,
    ) -> None:
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        pass

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        pass

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        pass
