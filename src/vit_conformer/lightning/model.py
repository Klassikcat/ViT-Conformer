from typing import *

from lightning.pytorch.utilities.types import STEP_OUTPUT, EVAL_DATALOADERS
from omegaconf import DictConfig
from pathlib import Path, PosixPath

import torch
from torch import nn, Tensor
import lightning.pytorch as ptl
from ..nn.conformer import ConformerBlock, VitConformerBlock, Linear, Conv2dSubsampling, Transpose


class ConformerBase(ptl.LightningModule):
    def __init__(
            self,
            num_vocab: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            joint_ctc_attention: bool = True,
    ) -> None:
        super().__init__()
        self.joint_ctc_attention = joint_ctc_attention
        self.input_projection = nn.Sequential(
            Linear(self.conv_subsample.get_output_dim(), encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )
        self.conv_subsample = Conv2dSubsampling(input_dim, in_channels=1, out_channels=encoder_dim)
        if self.joint_ctc_attention:
            self.fc = nn.Sequential(
                Transpose(shape=(1, 2)),
                nn.Dropout(feed_forward_dropout_p),
                Linear(encoder_dim, num_vocab, bias=False),
            )
        else:
            self.fc = Linear(encoder_dim, num_vocab, bias=False)
        self.layers = nn.ModuleList([])

    def set_beam_decoder(self, beam_size: int = 3):
        """Setting beam search decoder"""
        from ..decoders.beam_search import BeamSearchCTC

        self.decoder = BeamSearchCTC(
            labels=self.tokenizer.labels,
            blank_id=self.tokenizer.blank_id,
            beam_size=beam_size,
        )

    def collect_outputs(
            self,
            stage: str,
            logits: torch.FloatTensor,
            output_lengths: torch.IntTensor,
            targets: torch.IntTensor,
            target_lengths: torch.IntTensor,
    ) -> Dict[str, Tensor]:
        loss = self.criterion(
            log_probs=logits.transpose(0, 1),
            targets=targets[:, 1:],
            input_lengths=output_lengths,
            target_lengths=target_lengths,
        )
        predictions = logits.max(-1)[1]

        wer = self.wer_metric(targets[:, 1:], predictions)
        cer = self.cer_metric(targets[:, 1:], predictions)

        self.info(
            {
                f"{stage}_wer": wer,
                f"{stage}_cer": cer,
                f"{stage}_loss": loss,
                "learning_rate": self.get_lr(),
            }
        )
        return {"loss": loss, "wer": wer, "cer": cer}

    def forward(
            self,
            inputs: torch.Tensor,
            input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Forward propagate a `inputs` for  encoders training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor, Tensor)

            * outputs: A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
            * encoder_logits: Log probability of encoders outputs will be passed to CTC Loss.
                If joint_ctc_attention is False, return None.
            * output_lengths: The length of encoders outputs. ``(batch)``
        """
        encoder_logits = None

        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_projection(outputs)

        for layer in self.layers:
            outputs = layer(outputs)

        if self.joint_ctc_attention:
            encoder_logits = self.fc(outputs.transpose(1, 2)).log_softmax(dim=2)

        return outputs, encoder_logits, output_lengths

    def training_step(self, batch: tuple, batch_idx: int) -> Dict[str, Tensor]:
        r"""
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch
        encoder_outputs, encoder_logits, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(encoder_outputs).log_softmax(dim=-1)
        return self.collect_outputs(
            stage="train",
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> Dict[str, Tensor]:
        r"""
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch
        encoder_outputs, encoder_logits, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(encoder_outputs).log_softmax(dim=-1)
        return self.collect_outputs(
            stage="valid",
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def test_step(self, batch: tuple, batch_idx: int) -> Dict[str, Tensor]:
        r"""
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch
        encoder_outputs, encoder_logits, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(encoder_outputs).log_softmax(dim=-1)
        return self.collect_outputs(
            stage="test",
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )


class ConformerModel(ConformerBase):
    def __init__(
            self,
            num_vocab: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            joint_ctc_attention: bool = True,
    ) -> None:
        super().__init__(
            num_vocab,
            input_dim,
            encoder_dim,
            input_dropout_p,
            feed_forward_dropout_p,
            joint_ctc_attention,
        )
        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    half_step_residual=half_step_residual,
                )
                for _ in range(num_layers)
            ]
        )


class ViTConformerModel(ConformerBase):
    def __init__(
            self,
            num_vocab: int,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            joint_ctc_attention: bool = True,
    ) -> None:
        super().__init__(
            num_vocab,
            input_dim,
            encoder_dim,
            input_dropout_p,
            feed_forward_dropout_p,
            joint_ctc_attention,
        )
        self.layers = nn.ModuleList(
            [
                VitConformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    half_step_residual=half_step_residual,
                )
                for _ in range(num_layers)
            ]
        )

