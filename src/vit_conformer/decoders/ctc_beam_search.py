from typing import List, Dict
from torch import Tensor, nn


class BeamSearchCTC(nn.Module):
    r"""
    Decodes probability output using ctcdecode package.

    Args:
        labels (list): the tokens you used to train your model
        lm_path (str): the path to your external kenlm language model(LM).
        alpha (int): weighting associated with the LMs probabilities.
        beta (int): weight associated with the number of words within our beam
        cutoff_top_n (int): cutoff number in pruning. Only the top cutoff_top_n characters with the highest probability
            in the vocab will be used in beam search.
        cutoff_prob (float): cutoff probability in pruning. 1.0 means no pruning.
        beam_size (int): this controls how broad the beam search is.
        num_processes (int): parallelize the batch using num_processes workers.
        blank_id (int): this should be the index of the CTC blank token

    Inputs: logits, sizes
        - logits: Tensor of character probabilities, where probs[c,t] is the probability of character c at time t
        - sizes: Size of each sequence in the mini-batch

    Returns:
        - outputs: sequences of the model's best prediction
    """

    def __init__(
            self,
            labels: list,
            lm_path: str = None,
            alpha: int = 0,
            beta: int = 0,
            cutoff_top_n: int = 40,
            cutoff_prob: float = 1.0,
            beam_size: int = 3,
            num_processes: int = 4,
            blank_id: int = 0,
    ) -> None:
        super(BeamSearchCTC, self).__init__()
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError("ctcdecode is not installed. Please see the installation section for instructions.")
        assert isinstance(labels, list), "labels must instance of list"
        self.decoder = CTCBeamDecoder(
            labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_size, num_processes, blank_id
        )

    def forward(self, logits, sizes=None) -> Tensor:
        r"""
        Decodes probability output using ctcdecode package.

        Inputs: logits, sizes
            logits: Tensor of character probabilities, where probs[c,t] is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch

        Returns:
            outputs: sequences of the model's best prediction
        """
        logits = logits.cpu()
        outputs, scores, offsets, seq_lens = self.decoder.decode(logits, sizes)
        return outputs
