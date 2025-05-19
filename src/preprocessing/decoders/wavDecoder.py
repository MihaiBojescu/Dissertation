import io
import wave
import struct
import numpy as np
from interfaces.preprocessing.decoder import BaseDecoder
from interfaces.preprocessing.files import DecodedAudioFile, RawAudioFile


class WavDecoder(BaseDecoder):
    __fmt_char_map: dict[int, str]

    def __init__(self) -> None:
        super().__init__()
        self.__fmt_char_map = {1: "b", 2: "h", 4: "i"}

    def decode(self, raw_file: RawAudioFile) -> DecodedAudioFile:
        bio = io.BytesIO(raw_file.data)
        with wave.open(bio, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        fmt_char = self.__fmt_char_map[sampwidth]
        fmt = "<" + fmt_char * n_channels * n_frames
        all_samples = struct.unpack(fmt, raw_bytes)
        data = np.array(all_samples)

        return DecodedAudioFile(
            name=raw_file.name,
            extension=raw_file.extension,
            n_channels=n_channels,
            sample_rate=sample_rate,
            bits_per_sample=sampwidth * 8,
            data=data,
        )
