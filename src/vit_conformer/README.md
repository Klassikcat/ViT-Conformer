# Conformer with ViT
### Convolution-augmented Transformer for Speech Recognition + How do ViT Work?
## Introduction

This repository is based on [Conformer: Convolution-augmented Transformer for Speech Recognition(2020)](https://arxiv.org/abs/2005.08100) and
[How do Vision Transformer Work?](https://arxiv.org/abs/2202.06709). While original paper: Conformer: Convolution-augmented
Transformer for Speech Recognition states that combination of CNN network and Attention network enables the model to capture
both local and global features of speech, It turned out that the statement is not valid. Rather, combination of CNN network
and attention network flatten loss surface, so that the model finds global minima easier than CNN or Transformers during
the optimization process.

Now, I convert Conformer model from original concept to concept proposed on "How do Vision Transformer
work?" and see the statement on the new article is still valid in Speech Recognition.
