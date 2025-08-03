'''
This script is used to get the data from the Belgium signal db
'''

import os
import subprocess
from PIL import Image
import numpy as np


class Unpacker:

    def __init__(self) -> None:

        self.db_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..'))

        self.data_path = os.path.join(self.db_path, 'belgium-ts')

        # download the data
        self._download_data()

        # transform the data into a numpy object
        self._get_data()

    def _download_data(self) -> None:
        '''
        This method checks if the folder db_path already contains the data base
        and, in case it does not, will download it using a .sh script.
        '''

        os.chdir(self.db_path)

        # If the db does not exist we create one
        if not 'belgium-ts' in os.listdir():
            print('Downloading the data, please wait:\n')
            script_path = os.path.join(
                os.path.dirname(__file__), 'download-data.sh')
            subprocess.run(['bash', script_path], check=True)
            os.system('clear')

        os.chdir(os.path.join(self.db_path, 'belgium-ts'))

    def _get_data(self) -> None:
        '''
        This method will create some data sets for test and training from the db
        '''

        train = []
        train_label = []
        test = []
        test_label = []

        for folder in ['Training', 'Testing']:

            os.chdir('./BelgiumTSC_' + folder + '/' + folder)

            for file in os.listdir():
                if file.endswith('txt'):
                    continue

                label = int(file)
                os.chdir(file)

                for im in os.listdir():

                    if im.endswith('.csv'):
                        continue
                    im_pillow = Image.open(im).resize((150, 150))
                    im_arr = np.array(im_pillow) / 255.

                    if folder == 'Training':
                        train.append(im_arr)
                        train_label.append(label)

                    else:
                        test.append(im_arr)
                        test_label.append(label)

                os.chdir('..')

            os.chdir(self.db_path + '/belgium-ts')

        self.train = train
        self.test = test
        self.train_label = train_label
        self.test_label = test_label
