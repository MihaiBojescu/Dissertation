import numpy as np
from scipy import signal
from common.preprocess.files import (
    DecodedAudioFile,
    TransformedAudioFile,
    TransformedAudioFileChannelData,
)
from common.preprocess.transformer import BaseSpectrogramTransformer


class SpectrogramTransformer(BaseSpectrogramTransformer):
    def encode(
        self,
        decoded_file: DecodedAudioFile,
        sampling_rate: int,
        n_ffts: int,
        hop_length: int,
    ) -> TransformedAudioFile:
        window = np.blackman(n_ffts)
        data: list[TransformedAudioFileChannelData] = []

        sft = signal.ShortTimeFFT(
            win=window, fs=sampling_rate, hop=hop_length, mfft=n_ffts, scale_to="psd"
        )

        for channel in range(0, decoded_file.n_channels):
            channel_data = decoded_file.data[channel :: decoded_file.n_channels]
            channel_data = np.ascontiguousarray(channel_data)
            channel_data = channel_data.astype(np.float32) / np.max(
                np.abs(channel_data)
            )

            Sx = sft.stft(x=channel_data)

            magnitude = np.abs(Sx)
            num_time_bins = magnitude.shape[1]
            num_freq_bins = magnitude.shape[0]
            times = (np.arange(num_time_bins + 1) * hop_length) / sampling_rate
            frequencies = np.linspace(0, sampling_rate // 2, num_freq_bins + 1)

            data.append(
                TransformedAudioFileChannelData(
                    Sx=Sx,
                    times=times,
                    frequencies=frequencies,
                    magnitude=magnitude,
                )
            )

        return TransformedAudioFile(
            name=decoded_file.name,
            extension=decoded_file.extension,
            bits_per_sample=decoded_file.bits_per_sample,
            sample_rate=decoded_file.sample_rate,
            data=data,
        )

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

        n_channels = len(transformed_file.data)
        channel_lengths = [
            sft.istft(channel.Sx).shape[0] for channel in transformed_file.data
        ]
        max_length = max(channel_lengths)
        reconstructed = np.zeros((max_length, n_channels), dtype=np.float32)

        for channel_idx, channel_data in enumerate(transformed_file.data):
            window = np.blackman(channel_data.magnitude.shape[0])
            sft = signal.ShortTimeFFT(
                win=window,
                fs=transformed_file.sample_rate,
                hop=hop_length,
                mfft=channel_data.magnitude.shape[0],
                scale_to="psd",
            )

            reconstructed_channel = sft.istft(channel_data.Sx)

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
