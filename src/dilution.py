__author__ = "Adrián Pallas Fernández"
__email__ = "adrian.pallas@aimen.es"

import argparse
import training_tools
import os

DATASET_FOLDER = os.path.join(os.getcwd(),'ConvLMD_dataset')

def main():

    file_train = DATASET_FOLDER + os.sep +'dilution.hdf5'

    y = 'dilution'

    training_tools.leave_one_out(file_train, 'id_recording', 'image', y)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to de image")
    args = parser.parse_args()

    if args.path is not None:
        DATASET_FOLDER = args.path

    main()