import pandas as pd
from scipy import signal
from scipy.fftpack import fft


def break_2_bands(df):
    """
    Creates new df from input df after band passing each channel 5 times with 5 different frequency bands.
    Args:
        df: Pandas data frame with 2 EEG channels Fpz-Cz, Pz-Oz.

    Returns:
        df: Pandas data frame with 10 channels. 
    """
    delta = [0.4, 4]
    theta = [4, 8]
    alpha = [8, 13]
    sigma = [13, 22]
    beta = [30, 49.5]
    bands = [delta, theta, alpha, sigma, beta]
    
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
        channel_fft = fft(df.iloc[:, channel].values)
        intermidiate_dict[column_name] = channel_fft

    new_df = pd.DataFrame(data=intermidiate_dict)

    return new_df