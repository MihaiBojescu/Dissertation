import os
import matplotlib.pyplot as plt
from preprocess.spectrogramTransformer import SpectrogramTransformer
from preprocess.wavDecoder import WavDecoder
from common.preprocess.decoder import BaseDecoder
from common.preprocess.files import DecodedAudioFile, RawAudioFile, TransformedAudioFile
from common.preprocess.transformer import BaseSpectrogramTransformer


class Preprocessor:
    __input_path: str
    __output_path: str
    __decoders: dict[str, BaseDecoder]
    __transformer: BaseSpectrogramTransformer

    def __init__(
        self,
        input_path: str = "./data/input",
        output_path: str = "./data/spectrogram",
        decoders: dict[str, BaseDecoder] = {"wav": WavDecoder()},
        transformer: BaseSpectrogramTransformer = SpectrogramTransformer(),
    ) -> None:
        self.__input_path = input_path
        self.__output_path = output_path
        self.__decoders = decoders
        self.__transformer = transformer

    def run(
        self,
        n_ffts: int = 1024,
        hop_length: None | int = None,
    ):
        if hop_length is None:
            hop_length = n_ffts // 2

        for raw_file in self.__read_files():
            decoded_file = self.__decode(raw_file)
            transformed_file = self.__transform(decoded_file, n_ffts, hop_length)

            self.__write_file(transformed_file)

    def __read_files(self):
        for file_path in os.listdir(self.__input_path):
            full_file_path = f"{self.__input_path}/{file_path}"
            file_name, file_extension = os.path.splitext(file_path)
            file_extension = file_extension[1:]
            file_data = b""

            if not os.path.isfile(full_file_path):
                continue

            if not file_extension in self.__decoders:
                continue

            with open(full_file_path, "rb") as file:
                file_data = file.read()

            yield RawAudioFile(name=file_name, extension=file_extension, data=file_data)

    def __decode(self, raw_file: RawAudioFile) -> DecodedAudioFile:
        decoder = self.__decoders[raw_file.extension]
        return decoder.decode(raw_file)

    def __transform(
        self,
        decoded_file: DecodedAudioFile,
        n_ffts: int,
        hop_length: int,
    ) -> TransformedAudioFile:
        return self.__transformer.encode(
            decoded_file=decoded_file,
            sampling_rate=decoded_file.sample_rate,
            n_ffts=n_ffts,
            hop_length=hop_length,
        )

    def __write_file(self, transformed_file: TransformedAudioFile):
        path_prefix = (
            f"{self.__output_path}/{transformed_file.name}.{transformed_file.extension}"
        )
        plt.switch_backend("TkAgg")

        for i, entry in enumerate(transformed_file.data):
            time_bins = entry.magnitude.shape[1] / 1000
            freq_bins = entry.magnitude.shape[0] / 1000

            plt.figure(figsize=(time_bins, freq_bins))
            plt.pcolormesh(
                entry.times,
                entry.frequencies,
                entry.magnitude,
                shading="auto",
                cmap="viridis",
            )
            plt.ylim(0, transformed_file.sample_rate // 2)

            plt.axis("off")
            plt.title("")
            plt.xlabel("")
            plt.ylabel("")

            plt.savefig(
                f"{path_prefix}.{i}.png", bbox_inches="tight", pad_inches=0, dpi=10000
            )
            plt.close()
