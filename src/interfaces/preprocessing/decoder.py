from abc import ABC, abstractmethod
from interfaces.preprocessing.files import DecodedAudioFile, RawAudioFile


class BaseDecoder(ABC):
    extension: str

    @abstractmethod
    def decode(self, raw_file: RawAudioFile) -> DecodedAudioFile:
        pass
