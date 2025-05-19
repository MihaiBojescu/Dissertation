import csv
from multiprocessing.pool import AsyncResult
import os
import typing as t
import multiprocessing
import tqdm
from interfaces.preprocessing.files import RawAudioFile
from preprocessing.transformers.spectrogramTransformer import SpectrogramTransformer
from preprocessing.decoders.wavDecoder import WavDecoder
from interfaces.preprocessing.decoder import BaseDecoder
from interfaces.preprocessing.transformer import BaseSpectrogramTransformer
from preprocessing.worker import Worker


class Preprocessor:
    __input_path: str
    __output_path: str
    __decoders: dict[str, BaseDecoder]
    __transformer: BaseSpectrogramTransformer
    __n_procs: int

    def __init__(
        self,
        input_path: str = "./data/input",
        output_path: str = "./data/spectrogram",
        decoders: dict[str, BaseDecoder] = {"wav": WavDecoder()},
        transformer: BaseSpectrogramTransformer = SpectrogramTransformer(),
        n_procs: int = 2,
    ) -> None:
        self.__input_path = input_path
        self.__output_path = output_path
        self.__decoders = decoders
        self.__transformer = transformer
        self.__n_procs = n_procs

    def run(
        self,
        n_ffts: int = 1024,
        hop_length: None | int = None,
    ):
        results: list[AsyncResult] = []
        count = self.__count_files()

        with open(
            f"{self.__output_path}/dataset.csv", "w", encoding="utf-8"
        ) as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(("filename", "data", "channel"))
            csv_file.flush()
            worker = Worker(
                self.__output_path,
                self.__decoders,
                self.__transformer,
                n_ffts,
                hop_length,
            )

            with multiprocessing.Pool(processes=self.__n_procs) as pool:
                raw_files = self.__read_files()

                for raw_file in tqdm.tqdm(raw_files, total=count):
                    if len(results) > self.__n_procs:
                        results[0].wait()
                        results.pop(0)

                    result = pool.apply_async(
                        worker.run,
                        args=(raw_file,),
                        callback=self.__write(csv_writer, csv_file),
                    )
                    results.append(result)

            for result in results:
                result.wait()

    def __count_files(self):
        return len(os.listdir(self.__input_path))

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

    def __write(self, csv_writer: t.Any, csv_file: t.Any):
        def run(rows: list[tuple[str, str, int]]):
            csv_writer.writerows(rows)
            csv_file.flush()

        return run
