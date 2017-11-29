# -*- coding: utf-8 -*-
"""@author: carlos.gonzalez"""

from sklearn.metrics import mean_squared_error
import keras
import numpy as np
import os, csv, argparse

DATASET_FOLDER  = os.path.join(os.getcwd(),'DATASET')

class Data_generator:
    def __init__(self,path,files,BATCH_SIZE):
        self.files = files
        self.path=path
        self.BATCH_SIZE = BATCH_SIZE
        self.idx = None
        self.frames=None
        self.labels=None

    def count(self):
        return np.sum([len((np.load(os.path.join(self.path, d +'.npy')))[2000:-2000]) for d in self.files[:,0]])
        
    def all_data(self):
        frames=[self.__preprocess((np.load(os.path.join(self.path, d +'.npy')))[2000:-2000]) for d in self.files[:,0]]
        labels=[np.tile(self.files[i][1:],(len(frames[i]),1)) for i in range(len(frames))]
        return np.concatenate(frames),np.concatenate(labels)

    def __load_next_file(self):
        frames=np.load(os.path.join(self.path,self.files[self.idx][0]+'.npy'))[2000:-2000]
        self.labels = np.tile(self.files[self.idx][1:],(len(frames),1))
        self.frames=self.__preprocess(frames)
        if self.idx >= (len(self.files)-1):
            self.idx=0
        else:
            self.idx=self.idx+1
    
    def __preprocess(self,frames):
        return np.asarray(frames[:,2:30,2:30].reshape(frames.shape[0],28,28,1),np.float32)/1024.
    
    def get_batch(self):
        while True:
            if(self.frames is None):
                self.idx = 0
                self.__load_next_file()
            while len(self.frames)<self.BATCH_SIZE:
                self.__load_next_file()
                
            self.batch_x = self.frames[0:self.BATCH_SIZE]
            self.frames = np.delete(self.frames,np.s_[0:self.BATCH_SIZE],0)
            
            self.batch_y = self.labels[0:self.BATCH_SIZE].astype(np.float)
            self.labels = np.delete(self.labels,np.s_[0:self.BATCH_SIZE],0)
            
            yield self.batch_x, self.batch_y
        
        
def main():
    BATCH_SIZE = 128        # Number of training samples per iteration
    TRAINING_ITERS = 10000  # Number of training iterations
    OUTPUTS = 2             # Number of parameters for regression or classes for classification
    MODE = 'regression'     # Modes: 'regression' or 'classification'
    
    train_files , validation_files = load_indexes(DATASET_FOLDER)

    train_data = Data_generator(DATASET_FOLDER,train_files,BATCH_SIZE)  
    validation_data = Data_generator(DATASET_FOLDER,validation_files,BATCH_SIZE)  
    validation_iters = validation_data.count()//BATCH_SIZE    
    
    input_img = keras.layers.Input(shape=(28, 28, 1))
    conv_1 = keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(input_img)
    maxpool_1 = keras.layers.MaxPooling2D((2, 2))(conv_1)
    conv_2 = keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')(maxpool_1)
    maxpool_2 = keras.layers.MaxPooling2D((2, 2))(conv_2)
    out = keras.layers.Flatten()(maxpool_2)
    dense_1 = keras.layers.Dense(1024, activation='relu')(out)
    dropout_1=keras.layers.Dropout(0.75)(dense_1)
    
    if MODE == 'regression':
        dense_2 = keras.layers.Dense(OUTPUTS, activation='linear')(dropout_1)
    elif MODE == 'classification':
        dense_2 = keras.layers.Dense(OUTPUTS, activation='softmax')(dropout_1)
        
    model = keras.models.Model(inputs=input_img, outputs=dense_2)
    
    if MODE == 'classification':
        model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    elif MODE == 'regression':
        model.compile(optimizer='adam',
                  loss='mean_squared_error')
    
    model.fit_generator(generator = train_data.get_batch(),
                    steps_per_epoch = TRAINING_ITERS,
                    validation_data = validation_data.get_batch(),
                    validation_steps = validation_iters,
                    use_multiprocessing = True)
                    
    x_val,y_val=validation_data.all_data()
    
    score(y_val,model.predict(x_val))

def load_indexes(path, validation_samples=10, shuffle=True):
    csv_file = open(os.path.join(path,'cords.csv'), 'r')
    reader = csv.reader(csv_file,delimiter=' ',)
    index = np.array([[col for col in row] for row in reader])
    if shuffle:
        index=index[np.random.shuffle(np.arange(len(index)))][0]
    return index[:-validation_samples,:], index[-validation_samples:,:]
    
def score(y_true,y_pred):
    print('Mean square error of parameter 1 [Power]: ' + str(mean_squared_error(y_true[:,0].astype(np.float), y_pred[:,0].astype(np.float))))
    print('Mean square error of parameter 2 [Speed]: ' + str(mean_squared_error(y_true[:,1].astype(np.float), y_pred[:,1].astype(np.float))))
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to de image")
    args = parser.parse_args()

    DATASET_FOLDER = args.path

    main()