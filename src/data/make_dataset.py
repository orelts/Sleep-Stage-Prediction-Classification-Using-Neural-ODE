# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import mne
import matplotlib.pyplot as plt


# Data source and documentation https://physionet.org/content/sleep-edfx/1.0.0/
"""
For each subject we load data from 2 files. 
    1. EEG and sleeping records - PSG file.
    2. Annotation (labeling) - Hypnogram file.
    
For the EEG we have few channels recorded depends if its Sleep Telemetry or Sleep cassette.
"""


@click.command()
@click.argument('fname_psg', type=click.Path(exists=True))  # ST7011J0-PSG
@click.argument('fname_hyp', type=click.Path(exists=True))  # ST7011JP-Hypnogram
@click.argument('output_filepath', type=click.Path())
# TODO: load PSG and Hypnogram differently
def main(fname_psg, fname_hyp, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')
    logger.info(f"Project Directory is set to {project_dir}")

    logger.info(f'Trying to read PSG data file {fname_psg}')

    # Read actual data
    data_psg = mne.io.read_raw_edf(fname_psg, preload=True)
    logger.info(f'PSG data fetched {data_psg.info}')

    # Plot
    logger.info(f'Plotting PSG data {data_psg.info}')
    data_psg.plot(block=True)

    logger.info(f'Trying to read Hypnogram anotations file {fname_hyp}')
    data_hyp = mne.read_annotations(fname_hyp)

    # Creating annotiations dict
    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}

    # keep last 30-min wake events before sleep and first 30-min wake events after
    # sleep and redefine annotations on raw data
    data_hyp.crop(data_hyp[1]['onset'] - 30 * 60,
                     data_hyp[-2]['onset'] + 30 * 60)
    data_psg.set_annotations(data_hyp, emit_warning=False)

    events_train, _ = mne.events_from_annotations(
        data_psg, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    # create a new event_id that unifies stages 3 and 4
    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3/4': 4,
                'Sleep stage R': 5}

    # plot events
    fig = mne.viz.plot_events(events_train, event_id=event_id,
                              sfreq=data_psg.info['sfreq'],
                              first_samp=events_train[0, 0])

    # keep the color-code for further plotting
    stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
