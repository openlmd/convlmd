__author__ = "Adrián Pallas Fernández"
__email__ = "adrian.pallas@aimen.es"


import argparse
import training_tools
import os

DATASET_FOLDER = os.path.join(os.getcwd(),'ConvLMD_dataset')

def main():

    file_train = DATASET_FOLDER + os.sep +'train_pow_speed.hdf5'
    file_val = DATASET_FOLDER + os.sep +'test_pow_speed.hdf5'

    ys = ['power','velocity']

    training_tools.training(file_train,file_val,'id_recording','image',ys)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to de image")
    args = parser.parse_args()

    if args.path is not None:
        DATASET_FOLDER = args.path

    main()