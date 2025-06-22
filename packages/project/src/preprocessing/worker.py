import numpy as np
from preprocessing.transformers.spectrogramTransformer import SpectrogramTransformer
from preprocessing.decoders.wavDecoder import WavDecoder
from interfaces.preprocessing.decoder import BaseDecoder
from interfaces.preprocessing.files import DecodedAudioFile, RawAudioFile, TransformedAudioFile
from interfaces.preprocessing.transformer import BaseSpectrogramTransformer


class Worker:
    __output_path: str
    __decoders: dict[str, BaseDecoder]
    __transformer: BaseSpectrogramTransformer
    __n_ffts: int
    __hop_length: int

    def __init__(
        self,
        output_path: str = "./data/spectrogram",
        decoders: dict[str, BaseDecoder] = {"wav": WavDecoder()},
        transformer: BaseSpectrogramTransformer = SpectrogramTransformer(),
        n_ffts: int = 1024,
        hop_length: int | None = None,
    ):
        if hop_length is None:
            hop_length = n_ffts // 2

        self.__output_path = output_path
        self.__decoders = decoders
        self.__transformer = transformer
        self.__n_ffts = n_ffts
        self.__hop_length = hop_length

    def run(self, raw_file: RawAudioFile):
        decoded_file = self.__decode(raw_file)
        transformed_file = self.__transform(decoded_file)
        files = self.__write_file(transformed_file)

        return files

    def __decode(self, raw_file: RawAudioFile) -> DecodedAudioFile:
        decoder = self.__decoders[raw_file.extension]
        return decoder.decode(raw_file)

    def __transform(
        self,
        decoded_file: DecodedAudioFile,
    ) -> TransformedAudioFile:
        return self.__transformer.encode(
            decoded_file=decoded_file,
            sampling_rate=decoded_file.sample_rate,
            n_ffts=self.__n_ffts,
            hop_length=self.__hop_length,
        )

    def __write_file(
        self, transformed_file: TransformedAudioFile
    ) -> list[tuple[str, str, int]]:
        files: list[tuple[str, str, int]] = []
        file_name_prefix = f"{transformed_file.name}.{transformed_file.extension}"

        for i, entry in enumerate(transformed_file.data):
            with open(
                f"{self.__output_path}/{file_name_prefix}.{i}.npy", "wb"
            ) as out_file:
                np.save(out_file, entry.Sx)

            files.append(
                (
                    f"{transformed_file.name}.{transformed_file.extension}",
                    f"{file_name_prefix}.{i}.npy",
                    i,
                )
            )

        return files
