#!/usr/bin/env python3

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from interfaces.preprocessing.decoder import BaseDecoder
from preprocessing.preprocessor import Preprocessor
from preprocessing.spectrogramTransformer import SpectrogramTransformer
from preprocessing.wavDecoder import WavDecoder


def main():
    decoders: dict[str, BaseDecoder] = {"wav": WavDecoder()}
    transformer = SpectrogramTransformer()
    preprocessor = Preprocessor(decoders=decoders, transformer=transformer, n_procs=4)

    preprocessor.run(n_ffts=8192, hop_length=4096)


if __name__ == "__main__":
    main()
