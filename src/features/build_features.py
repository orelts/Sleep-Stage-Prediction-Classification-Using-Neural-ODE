import pandas as pd
from scipy import signal


def break_2_bands(df):
    bands = [[0.4, 4], [4, 8], [8, 13], [13, 22], [30, 49.5]]
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
