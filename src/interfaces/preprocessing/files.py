from dataclasses import dataclass
import numpy as np


@dataclass
class RawAudioFile:
    name: str
    extension: str
    data: bytes


@dataclass
class DecodedAudioFile(RawAudioFile):
    n_channels: int
    sample_rate: int
    bits_per_sample: int


@dataclass
class TransformedAudioFileChannelData:
    Sx: np.ndarray[tuple[int, ...], np.dtype[np.float16]]


@dataclass
class TransformedAudioFile:
    name: str
    extension: str
    sample_rate: int
    bits_per_sample: int
    data: list[TransformedAudioFileChannelData]
