# !/usr/bin/python3

import subprocess
import sys

DATA_DIR = r'data/prcoessed(75_subjects-train-test)/train'
TEST_RATIO = '0.2'
EPOCHS = '80'

for i in [4, 5]:

    subprocess.call([sys.executable, '-m',  'src.models.train_model',
                     '--data_dir', DATA_DIR,
                     '--ratio', TEST_RATIO,
                     '--nrof_files', ' 75',
                     '--save', f'logs/mixed_exp_{i}/',
                     '--log_name', f'mixed_exp_{i}',
                     '--nepochs', EPOCHS,
                     '--adjoint', ' 1'],
                    shell=False)
