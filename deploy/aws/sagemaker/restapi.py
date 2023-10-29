from dataclasses import dataclass


@dataclass
class RequestForm(object):
    audio_io: bytes
    sample_rate: int
    content_type: str


@dataclass
class ResponseForm(object):
    text: str
    status_code: int
    content_type: str

