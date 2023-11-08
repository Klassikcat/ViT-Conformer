from typing import Any, List

from mlflow import MlflowClient

import torch
from torch import Tensor
import lightning.pytorch as ptl


class PerformanceMonitoringModule:
    def __init__(
            self,
            labels: List[str],
            mlflow_client_tracking_uri: str,
    ) -> None:
        super().__init__()
        self.labels = labels
        self.tracking_client = MlflowClient(tracking_uri=mlflow_client_tracking_uri)


class PerformanceMonitoringBatchModule(PerformanceMonitoringModule, ptl.callbacks.Callback):
    def __init__(
            self,
            labels: List[str],
            mlflow_client_tracking_uri: str,
    ) -> None:
        super().__init__(
            labels=labels,
            mlflow_client_tracking_uri=mlflow_client_tracking_uri,
        )

    def on_predict_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pass

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass


class PerformanceMonitoringRealTimeModule(PerformanceMonitoringModule):
    def __init__(
            self,
            labels: List[str],
            mlflow_client_tracking_uri: str,
    ) -> None:
       super().__init__(
           labels=labels,
           mlflow_client_tracking_uri=mlflow_client_tracking_uri,
       )
