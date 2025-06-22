from abc import ABC, abstractmethod
from interfaces.preprocessing.files import DecodedAudioFile, TransformedAudioFile


class BaseSpectrogramTransformer(ABC):
    @abstractmethod
    def encode(
        self,
        decoded_file: DecodedAudioFile,
        sampling_rate: int,
        n_ffts: int,
        hop_length: int,
    ) -> TransformedAudioFile:
        pass

    @abstractmethod
    def decode(
        self,
        transformed_file: TransformedAudioFile,
        sampling_rate: int,
        n_ffts: int,
        hop_length: int,
    ) -> DecodedAudioFile:
        pass
