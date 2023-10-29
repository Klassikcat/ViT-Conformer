import requests
from typing import Optional
from pathlib import Path, PosixPath

import gradio


class GradioApp:
    def __init__(
            self,
            root_url: PosixPath,
            inference_dirpath: PosixPath,
            train_dirpath: Optional[PosixPath] = None,
            port: Optional[int] = None
    ) -> None:
        self.root_url = root_url
        self.inference_dirpath = inference_dirpath
        self.train_dirpath = train_dirpath
        self.port = port

    @property
    def inference_full_url(self) -> PosixPath:
        return Path(f"{self.root_url}:{self.port}") / self.inference_dirpath if self.port else self.root_url / self.inference_dirpath

    @property
    def train_full_url(self) -> PosixPath:
        if not self.train_dirpath:
            raise ValueError("train_dirpath is not defined")
        return Path(f"{self.root_url}:{self.port}") / self.train_dirpath if self.port else self.root_url / self.train_dirpath

    def run_asr_train(self, data_path: str):
        requests.post(url=str(self.inference_full_url), files={"data_path": data_path})

    def run_audio_inference(self, audio_io: bytes):
        return requests.post(url=str(self.train_full_url), files={"audio": audio_io})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_url', type=str, default="http://localhost")
    parser.add_argument('--inference_dirpath', type=str, default="inference")
    parser.add_argument('--train_dirpath', type=str, required=False, default=None)
    parser.add_argument('--port', type=int, required=False, default=None)
    args = parser.parse_args()

    frontend = GradioApp(
        root_url=args.root_url,
        inference_dirpath=args.inference_dirpath,
        train_dirpath=args.train_dirpath,
        port=args.port
    )

    gr = gradio.TabbedInterface(
        [frontend.run_audio_inference, frontend.run_asr_train],
        title="ASR",
        description="Automatic Speech Recognition",
        tab_names=['Inference', 'Train']
    )
    gr.launch()