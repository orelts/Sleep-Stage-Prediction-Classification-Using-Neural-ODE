# -*- coding: utf-8 -*-

import logging
from pathlib import Path
# from dotenv import find_dotenv, load_dotenv

from mne.datasets.sleep_physionet.age import fetch_data
import mne
import numpy as np
from datetime import datetime
import math
import ntpath
import os
import argparse
import src.utils as utils
from src.data import dhedfreader
from src.features.build_features import break_2_bands, channels_fft, eeg_power_band, prepare_epcohs

parser = argparse.ArgumentParser()
parser.add_argument('--pre_processing', type=int, default=0, choices=[0, 1, 2, 3], help="0-no preprocessing, "
                                                                                        "1-brake to bands, "
                                                                                        "2-FFT, "
                                                                                        "3-PSD")
parser.add_argument('--nrof_subjects', type=int, default=15)
parser.add_argument('--select_ch', nargs='*', default=['EEG Fpz-Cz', 'EEG Pz-Oz'])
parser.add_argument('--output_filepath', type=str, default=r'../../data/processed/')
parser.add_argument('--subjects_to_folders', default=True, choices=[True, False])
parser.add_argument('--verbose', type=int, default=logging.INFO)

args = parser.parse_args()

# Data source and documentation https://physionet.org/content/sleep-edfx/1.0.0/
"""
For each subject we load data from 2 files. 
    1. EEG and sleeping records - PSG file.
    2. Annotation (labeling) - Hypnogram file.
    
For the EEG we have few channels recorded depends if its Sleep Telemetry or Sleep cassette.
"""

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def preprocess(raw_edf):
    # Low Pass
    high_cut_off_hz = 30.

    # raw_edf.plot_psd(area_mode='range', tmax=10.0, average=False)
    raw_edf.filter(None, high_cut_off_hz, fir_design='firwin')
    # raw_edf.plot_psd(area_mode='range', tmax=10.0, average=False)

    return raw_edf


def prepare_physionet_files(files, output_dir, select_ch):
    logger = logging.getLogger(__name__)
    file_dir = 0
    for file in files:
        file_dir = file_dir + 1
        psg = file[0]
        anno = file[1]
        filename = ntpath.basename(psg).replace("-PSG.edf", ".npz")
        logger.info(f"Preparing Subject {file}")

        # Get raw header
        f = open(psg, 'r', errors='ignore')
        reader_raw = dhedfreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
        f.close()

        # Read annotation and its header
        f = open(anno, 'r', errors='ignore')
        reader_ann = dhedfreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header

        raw = mne.io.read_raw_edf(psg, preload=True, stim_channel=None)

        sampling_rate = raw.info['sfreq']
        logger.debug(raw.info)
        df = raw.to_data_frame(scalings=100.0)[select_ch]

        # Preprocessing
        if args.pre_processing == 1:
            df = break_2_bands(df)
        elif args.pre_processing == 2:
            df = channels_fft(df)
        elif args.preprocessing == 3:
            epcohs = prepare_epcohs(raw=raw, anno=anno, select_ch=select_ch)
            x, y = eeg_power_band(epochs=epcohs)

        if args.preprocessing != 3:
            logger.debug(df.head())
            df.set_index(np.arange(len(df)))

            raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

            _, _, ann = zip(*reader_ann.records())
            f.close()
            ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")

            # Assert that raw and annotation files start at the same time
            assert raw_start_dt == ann_start_dt

            # Generate label and remove indices
            remove_idx = []  # indicies of the data that will be removed
            labels = []  # indicies of the data that have labels
            label_idx = []
            for a in ann[0]:
                onset_sec, duration_sec, ann_char = a
                ann_str = "".join(ann_char)
                label = ann2label[ann_str[2:-1]]
                if label != UNKNOWN:
                    if duration_sec % EPOCH_SEC_SIZE != 0:
                        raise Exception("Something wrong")
                    duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
                    label_epoch = np.ones(duration_epoch, dtype=int) * label
                    labels.append(label_epoch)
                    idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
                    label_idx.append(idx)

                    logger.debug("Include onset:{}, duration:{}, label:{} ({})".format(
                        onset_sec, duration_sec, label, ann_str
                    ))
                else:
                    idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
                    remove_idx.append(idx)

                    logger.debug("Remove onset:{}, duration:{}, label:{} ({})".format(
                        onset_sec, duration_sec, label, ann_str))
            labels = np.hstack(labels)

            logger.debug("before remove unwanted: {}".format(np.arange(len(df)).shape))
            if len(remove_idx) > 0:
                remove_idx = np.hstack(remove_idx)
                select_idx = np.setdiff1d(np.arange(len(df)), remove_idx)
            else:
                select_idx = np.arange(len(df))
            logger.debug("after remove unwanted: {}".format(select_idx.shape))

            # Select only the data with labels
            logger.debug("before intersect label: {}".format(select_idx.shape))
            label_idx = np.hstack(label_idx)
            select_idx = np.intersect1d(select_idx, label_idx)
            logger.debug("after intersect label: {}".format(select_idx.shape))

            # Remove extra index
            if len(label_idx) > len(select_idx):
                logger.debug("before remove extra labels: {}, {}".format(select_idx.shape, labels.shape))
                extra_idx = np.setdiff1d(label_idx, select_idx)
                # Trim the tail
                if np.all(extra_idx > select_idx[-1]):
                    # n_trims = len(select_idx) % int(EPOCH_SEC_SIZE * sampling_rate)
                    # n_label_trims = int(math.ceil(n_trims / (EPOCH_SEC_SIZE * sampling_rate)))
                    n_label_trims = int(math.ceil(len(extra_idx) / (EPOCH_SEC_SIZE * sampling_rate)))
                    if n_label_trims != 0:
                        # select_idx = select_idx[:-n_trims]
                        labels = labels[:-n_label_trims]
                logger.debug("after remove extra labels: {}, {}".format(select_idx.shape, labels.shape))

            # Remove movement and unknown stages if any
            raw_ch = df.values[select_idx]

            # Verify that we can split into 30-s epochs
            if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
                raise Exception("Something wrong")
            n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * sampling_rate)

            # Get epochs and their corresponding labels
            x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
            y = labels.astype(np.int32)

            assert len(x) == len(y)

            # Select on sleep periods
            w_edge_mins = 30
            nw_idx = np.where(y != stage_dict["W"])[0]
            start_idx = nw_idx[0] - (w_edge_mins * 2)
            end_idx = nw_idx[-1] + (w_edge_mins * 2)
            if start_idx < 0: start_idx = 0
            if end_idx >= len(y): end_idx = len(y) - 1
            select_idx = np.arange(start_idx, end_idx + 1)
            logger.debug("Data before selection: {}, {}".format(x.shape, y.shape))
            x = x[select_idx]
            y = y[select_idx]
            logger.debug("Data after selection: {}, {}".format(x.shape, y.shape))

        # Save

        save_dict = {
            "x": x,
            "y": y,
            "fs": sampling_rate,
            "ch_label": select_ch,
            "header_raw": h_raw,
            "header_annotation": h_ann,
        }
        logger.info(f"Final Data Shape {x.shape}")
        logger.info(f"Final Data ch_label {select_ch}")
        if args.subjects_to_folders:
            makedirs(output_dir + f"{file_dir}")
            logger.info(f"Saving data into  {output_dir}{file_dir}")
            np.savez(os.path.join(output_dir + f"{file_dir}", filename), **save_dict)
        else:
            makedirs(output_dir)
            logger.info(f"Saving data into  {output_dir}")
            np.savez(os.path.join(output_dir, filename), **save_dict)


def prepare_physionet(files_train, files_test, output_train_dir, output_test_dir, select_ch="Fpz-Cz"):
    prepare_physionet_files(files=files_train, output_dir=output_train_dir, select_ch=select_ch)
    prepare_physionet_files(files=files_test, output_dir=output_test_dir, select_ch=select_ch)


def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # Logger config

    logger = logging.getLogger(__name__)
    logger.setLevel(args.verbose)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(args.verbose)
    ch.setFormatter(utils.CustomFormatter())
    logger.addHandler(ch)

    # Fetching subjects, there are few unknown subjects that their indices need to be removed from fetching list.
    subjects = list(range(args.nrof_subjects))
    unknown_subjects = [36, 39, 52, 68, 69, 78, 79]
    for unknown in unknown_subjects:
        if unknown in subjects:
            subjects.remove(unknown)

    logger.info(f'Fetching  subjects {subjects} from physionet dataset ')

    files = fetch_data(subjects=subjects, recording=[1])
    logger.info(f'OUTPUT_DIR:{output_filepath}')

    prepare_physionet_files(files=files, output_dir=output_filepath, select_ch=args.select_ch)


if __name__ == '__main__':
    makedirs(args.output_filepath)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main(args.output_filepath)
