from abc import ABC, abstractmethod
from common.preprocess.files import DecodedAudioFile, RawAudioFile


class BaseDecoder(ABC):
    extension: str

    @abstractmethod
    def decode(self, raw_file: RawAudioFile) -> DecodedAudioFile:
        pass
