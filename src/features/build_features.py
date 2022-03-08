import pandas as pd
from scipy import signal
from scipy.fftpack import fft
from mne.time_frequency import psd_welch
import numpy as np

delta = [0.4, 4]
theta = [4, 8]
alpha = [8, 11.5]
sigma = [11.5, 15.5]
beta = [15.5, 30]
bands = [delta, theta, alpha, sigma, beta]


def break_2_bands(df):
    """
    Creates new df from input df after band passing each channel 5 times with 5 different frequency bands.
    Args:
        df: Pandas data frame with 2 EEG channels Fpz-Cz, Pz-Oz.

    Returns:
        df: Pandas data frame with 10 channels. 
    """

    num_of_channels = df.ndim
    intermidiate_dict = {}
    for i in range(num_of_channels):
        for j, freq_band in enumerate(bands):
            sos = signal.butter(6, freq_band, 'bp', fs=100, output='sos')
            filtered = signal.sosfilt(sos, df[df.columns[i]].values)
            coloumn_name = f"{df.columns[i]}_band{j + 1}"
            intermidiate_dict[coloumn_name] = filtered

    new_df = pd.DataFrame(data=intermidiate_dict)

    return new_df


def channels_fft(df):
    """
    FFT on input df channels
    Args:
     df: Pandas data frame with 2 EEG channels Fpz-Cz, Pz-Oz.

    Returns:
        df with channels after fft (FFTed Fpz-Cz, FFTed Pz-Oz.)
    """
    num_of_channels = df.ndim
    intermidiate_dict = {}
    for channel in range(num_of_channels):
        column_name = f"{df.columns[channel]}_fft"
        sos = signal.butter(6, Wn=49, fs=100, output='sos')
        filtered = signal.sosfilt(sos, df.iloc[:, channel].values)
        channel_fft = abs(fft(filtered))
        intermidiate_dict[column_name] = channel_fft

    new_df = pd.DataFrame(data=intermidiate_dict)

    return new_df


def eeg_power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": delta,
                  "theta": theta,
                  "alpha": alpha,
                  "sigma": sigma,
                  "beta": beta}

    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=30.)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)