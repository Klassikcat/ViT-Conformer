import io
import logging
from typing import Union, Tuple
from pathlib import Path, PosixPath

import torch
import torch.neuron as neuron


logger = logging.getLogger(__name__)


def compile(
        checkpoint_io_or_path: Union[PosixPath, io.BytesIO],
        input_shape: Tuple[int, int, int],
        save_path: Union[PosixPath, str]
) -> None:
    model = torch.load(checkpoint_io_or_path)
    logger.info("analyzing models... \nmodel structure: ".format(model.eval()))
    neuron.analyze_model(model, example_inputs=[torch.zeros(input_shape)])
    neuron_model = torch.neuron.trace(model, example_inputs=[torch.zeros(input_shape)])
    neuron_model.save(save_path)
    logger.info("model saved to {}".format(save_path))

