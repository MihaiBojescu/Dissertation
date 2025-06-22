import numpy as np
from scipy import signal
from interfaces.preprocessing.files import (
    DecodedAudioFile,
    TransformedAudioFile,
    TransformedAudioFileChannelData,
)
from interfaces.preprocessing.transformer import BaseSpectrogramTransformer


class SpectrogramTransformer(BaseSpectrogramTransformer):
    def encode(
        self,
        decoded_file: DecodedAudioFile,
        sampling_rate: int,
        n_ffts: int,
        hop_length: int,
    ) -> TransformedAudioFile:
        window = np.blackman(n_ffts)

        sft = signal.ShortTimeFFT(
            win=window,
            fs=sampling_rate,
            hop=hop_length,
            mfft=n_ffts,
            scale_to="magnitude",
        )

        Sx = sft.stft(x=decoded_file.data)
        Sx = self.__to_decibels(Sx)
        Sx = np.stack([Sx.real, Sx.imag], axis=0).astype(np.float16)

        return TransformedAudioFile(
            name=decoded_file.name,
            extension=decoded_file.extension,
            bits_per_sample=decoded_file.bits_per_sample,
            sample_rate=decoded_file.sample_rate,
            data=[TransformedAudioFileChannelData(Sx=Sx)],
        )

    def __to_decibels(self, Sx: np.ndarray[tuple[int, ...], np.dtype[np.float32]]):
        eps = 1e-10
        Sx_db = 10 * np.log10(Sx + eps)
        return Sx_db

    def decode(
        self,
        transformed_file: TransformedAudioFile,
        sampling_rate: int,
        n_ffts: int,
        hop_length: int,
    ) -> DecodedAudioFile:
        window = np.blackman(n_ffts)
        sft = signal.ShortTimeFFT(
            win=window, fs=sampling_rate, hop=hop_length, mfft=n_ffts, scale_to="psd"
        )
        data = transformed_file.data.copy()

        for channel in data:
            channel.Sx = self.__from_decibels(channel.Sx)

        n_channels = len(transformed_file.data)
        channel_lengths = [sft.istft(channel.Sx).shape[0] for channel in data]
        max_length = max(channel_lengths)
        reconstructed = np.zeros((max_length, n_channels), dtype=np.float32)

        for channel_idx, channel_data in enumerate(data):
            window = np.blackman(channel_data.magnitude.shape[0])
            sft = signal.ShortTimeFFT(
                win=window,
                fs=transformed_file.sample_rate,
                hop=hop_length,
                mfft=channel_data.magnitude.shape[0],
                scale_to="psd",
            )

            reconstructed_channel = sft.istft(self.__from_decibels(channel_data.Sx))

            reconstructed_channel = reconstructed_channel / np.max(
                np.abs(reconstructed_channel)
            )
            reconstructed[: len(reconstructed_channel), channel_idx] = (
                reconstructed_channel
            )

        return DecodedAudioFile(
            name=transformed_file.name,
            extension=transformed_file.extension,
            n_channels=n_channels,
            sample_rate=transformed_file.sample_rate,
            bits_per_sample=transformed_file.bits_per_sample,
            data=reconstructed.tobytes(),
        )

    def __from_decibels(self, Sx_db: np.ndarray[tuple[int, ...], np.dtype[np.float32]]):
        eps = 1e-10
        Sx = 10 ** ((Sx_db + eps) / 10)
        return Sx
