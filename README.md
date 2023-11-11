# STT Pipeline with Vit-Conformer and AutoTrain
#### ViT with AutoTrain + Deployment + Local GUI Demo + ONNX + TensorRT

## Introduction


## Architecture

## Installation

See [documentation](./docs) for more details. Below instructions are for quick start.

<details onclose>
<summary>PyPI</summary>

```shell
git clone https://github.com/Klassikcat/ViT-Conformer
python3 pip install -U VitConformer # This installs both VitConformer and AutoTrain.
```
</details>

<details onclose>
<summary>Docker</summary>

```shell
git clone https://github.com/Klassikcat/ViT-Conformer
docker build ViT-Conformer/dockerFiles -t local.DockerFile
```
</details>

## Validate installation

```shell
python3 test/test_conformer_installation.py
python3 test/test_autotrain_installation.py
```

## Deploy to your server

<details open>
<summary>Deploy to AWS Server</summary>
 </details>

<details open>
<summary>Deploy to your local machine or on-premise server</summary>
</details>

## License

### Code

### Pre-trained Weights

## Citations

- Conformer modeling files: [OpenSpeech in OpenSpeech Team](https://github.com/openspeech-team/openspeech)
- CTC Beam Search: [CTCDecode of Parlance](https://github.com/parlance/ctcdecode)
- uses [ChatGPT](https://chat.openai.com) for Demo Front Page