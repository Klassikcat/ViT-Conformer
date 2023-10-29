# Conformer with ViT
### Convolution-augmented Transformer for Speech Recognition + How do ViT Work?
#### ViT with AutoTrain + Deployment + Local GUI Demo + ONNX + TensorRT

## Introduction

This repository is based on [Conformer: Convolution-augmented Transformer for Speech Recognition(2020)](https://arxiv.org/abs/2005.08100) and 
[How do Vision Transformer Work?](https://arxiv.org/abs/2202.06709). While original paper: Conformer: Convolution-augmented
Transformer for Speech Recognition states that combination of CNN network and Attention network enables the model to capture
both local and global features of speech, It turned out that the statement is not valid. Rather, combination of CNN network
and attention network flatten loss surface, so that the model finds global minima easier than CNN or Transformers during
the optimization process.

Now, I convert Conformer model from original concept to concept proposed on "How do Vision Transformer
work?" and see the statement on the new article is still valid in Speech Recognition.

## AutoTrain

The AutoTrain package is a powerful Python module designed to streamline the machine learning model training process.
It automates the monitoring of model performance, initiates retraining when performance decreases, and selects datasets
intelligently using state-of-the-art STT (Speech-to-Text) models. 

### Features
#### 1. Automatic Model Performance Detection
AutoTrain includes an intelligent mechanism to continuously monitor your model's performance. It evaluates the model's
confidence in its predictions and detects performance decreases. When the confidence falls below a certain threshold,
AutoTrain triggers the retraining process.

#### 2. Automatic Model Re-Train
When the model's performance declines, AutoTrain automatically initiates the retraining process. This feature ensures
that your machine learning model remains up-to-date and performs optimally.

#### 3. Automatically Select Datasets
AutoTrain makes dataset selection easier by leveraging STT models. These models can transcribe speech into text,
allowing you to choose datasets based on the content and context of the spoken language.
This dynamic dataset selection enhances your model's adaptability to specific domains or languages.



## Architecture

## Installation

See [documentation](./docs) for more details. Below instructions are for quick start.

### PyPI

```shell
git clone 
python3 pip install -U VitConformer # This installs both VitConformer and AutoTrain.
```

### Docker

```shell
docker build 
```

## Validate installation

```shell
python3 test/test_conformer_installation.py
python3 test/test_autotrain_installation.py
```

## Deploy to your server

### Deploy to AWS Server

### Deploy to your local machine or on-premise server.

## License

## Citations