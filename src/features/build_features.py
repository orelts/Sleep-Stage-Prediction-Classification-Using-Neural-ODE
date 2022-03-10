import pandas as pd
from scipy import signal
from scipy.fftpack import fft
from mne.time_frequency import psd_welch
import numpy as np
import mne

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


def prepare_epcohs(raw, anno, select_ch=None):

    if select_ch is None:
        select_ch = ["EEG Fpz-Cz", "EEG Pz-Oz"]

    sampling_rate = raw.info['sfreq']
    annot = mne.read_annotations(anno)

    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}

    # keep last 30-min wake events before sleep and first 30-min wake events after
    # sleep and redefine annotations on raw data
    annot.crop(annot[1]['onset'] - 30 * 60,
               annot[-2]['onset'] + 30 * 60)
    raw.set_annotations(annot, emit_warning=False)

    events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    # create a new event_id that unifies stages 3 and 4
    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3/4': 4,
                'Sleep stage R': 5}

    # Create Epochs from the data based on the events found in the annotations
    tmax = 30. - 1. / sampling_rate  # tmax in included

    epochs = mne.Epochs(raw=raw, events=events, preload=True,
                        event_id=event_id, tmin=0., tmax=tmax, baseline=None)
    epochs_eeg = epochs.pick_channels(select_ch)

    return epochs_eeg


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
        # Average over the frequencies of each band.
        psds_band = psds.copy()
        psds_band[:, :, (freqs < fmin) | (freqs > fmax)] = 0
        X.append(psds_band)

    X = np.concatenate(X, axis=1)
    X = X.transpose(0, 2, 1)
    y = epochs.events[:, 2] - 1

    return X, y