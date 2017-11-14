# -*- coding: utf-8 -*-


############################### TRAINING MODULE ###############################
#                                                                             #
# Training module for LasLearn                                                #
# Created on Fri Oct  6 13:23:49 2017                                         #
#                                                                             #
# Author: Carlos Gonzalez Val                                                 #
#                                                                             #
# Description: This module computes ML algorithms for a (X,y) dataset and     #
# outputs the relevant information in a text log.                             #
#                                                                             #
###############################################################################


from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import (manifold, decomposition, ensemble,lda)
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.decomposition import PCA
import pickle
import cv2
import logging
import tensorflow as tf
from PyQt5 import QtWidgets
            

class Training():
    def __init__(self,textLog,loggerHandler,loggingLevel, parent=None):
        self.textLog=textLog
        self.n_neighbors=30
        self.model=None
        self.predictor_loaded=False
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(loggerHandler) 
        self.logger.setLevel(loggingLevel)
        self.logger.info('Training module initiated')
        hdlr_tf=self.tfLogger(self.log)
        hdlr_tf.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        #tf.logging.set_verbosity(tf.logging.INFO)
        tf.logging._logger.propagate = True
        tf.logging._logger.addHandler(hdlr_tf) 

    class tfLogger(logging.Handler):
        def __init__(self,loggerf):
            logging.Handler.__init__(self)
            self.log=loggerf
        def emit(self, record):
            record=self.format(record)
            self.log('Info TF: ' + str(record))

    #Write to the log panel in the GUI
    def log(self,text):
        self.textLog.append(str(text))
        self.logger.info(text)
    
    #Call to the algorithm for data visualization
    def visualize(self,algorithm,data):
        X_images,X,y = data
        if algorithm =='PCA':
            self.PCA_proj(X,y)
        elif algorithm =='Random Trees':
            self.RT_proj(X,y)
        elif algorithm =='LDA':
            self.LDA_proj(X,y)
        elif algorithm =='Isomap':
            self.isomap_proj(X,y)
        elif algorithm =='LLE':
            self.LLE_proj(X,y)
        elif algorithm =='MLLE':
            self.MLLE_proj(X,y)
        elif algorithm =='HLLE':
            self.HLLE_proj(X,y)
        elif algorithm =='LTSA':
            self.LTSA_proj(X,y)
        elif algorithm =='MDS':
            self.MDS_proj(X,y)
        elif algorithm =='CSE':
            self.CSE_proj(X,y)
        elif algorithm =='t-SNE': #Not implemented
            self.tsne_proj(X,y)
        else:
            self.log(algorithm + ' algorithm is not implemented yet')
            
    #Call to the algorithm for training        
    def train(self,algorithm,data):
        self.predictor_loaded=False
        X_images,X,y = data
        if algorithm =='RT + SVM':
            status=self.RT_and_SVM(X,y)
        elif algorithm =='PCA + SVM':
            status=self.PCA_and_SVM(X,y)
        elif algorithm =='NN Conv 2 layers':
            status=self.NN_classifier(X_images,y)
        else:
            self.log('This algorithm is not implemented yet')
        return status

    #Call to the loaded model for evaluation and writes the output in the log
    def evaluate(self,data):
        try:
            X_images,X,y = data
            X=X[y!=-1]
            y=y[y!=-1]
            X_images=X_images[y!=-1]
            if self.model is not None:
                if type(self.model)=='sklearn.pipeline.Pipeline':
                    self.log("Score: " + str(self.model.score(X,y)))
                elif type(self.model)==tf.estimator.Estimator:
                    X=(np.array([cv2.resize(x_i,(32,32)) for x_i in X_images])[:,2:30,2:30])
                    evaluate_input_fn = tf.estimator.inputs.numpy_input_fn(
                        x={'x': X.astype(np.float32)/self.max},
                        y=y.astype(np.int32),
                        num_epochs=1,
                        shuffle=False)
                    scores = self.model.evaluate(input_fn=evaluate_input_fn)
                    self.log('Accuracy of the model: {0:f}'.format(scores['accuracy']))
                elif self.predictor_loaded:
                    ytest=self.predict(X_images)
                    y=y.astype(np.int32)
                    self.log("Score: " + str(float(sum(y==ytest))/len(y)*100))
                else:
                    self.log("Error during evaluation. Check log for more details.")
                    self.logger.info(type(self.model))
        except Exception,e:
            self.logger.warning(str(e))
            self.logger.warning('Error during evaluation.')
            self.log("Error during evaluation. Check log for more details.")
           
    #Call to the loaded model for prediction and return a vector of labels
    def predict(self,X):
        try:
            if self.model is not None:
                if type(self.model)==Pipeline:
                    m, h, w = np.shape(X)
                    X=X.reshape(m,h*w)
                    y=self.model.predict(X)
                elif type(self.model)==tf.estimator.Estimator:
                    X=(np.array([cv2.resize(x_i,(32,32)) for x_i in X])[:,2:30,2:30])
                    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                        x={'x': X.astype(np.float32)/self.max},
                        num_epochs=1,
                        shuffle=False)
                    predictions = self.model.predict(input_fn=predict_input_fn)
                    y=[]
                    for i, p in enumerate(predictions):
                        y.append(p['class'])
                elif self.predictor_loaded:
                    self.max=np.max(X)
                    x=(np.array([cv2.resize(x_i,(32,32)) for x_i in X])[:,2:30,2:30]).astype(np.float32)/self.max
                    m, h, w = np.shape(x)
                    x=x.reshape(m,h*w)
                    predictions = self.model({'x':x})
                    y=predictions['y']
                else:
                    raise
                return y
        except Exception,e:
            self.logger.warning(str(e))
            self.logger.warning('Error during prediction.')
            self.log("Error during prediction. Check log for more details.")
    
    #Loads a trained model        
    def load_model(self, parent,dirname):
        try:
            filename,_ = QtWidgets.QFileDialog.getOpenFileName(
                parent, 'Open model',dirname, 'Pickle Files (*.pkl);;Tensor Flow (*.pb)')
            filename=str(filename)    
            typeoffile=filename[(filename.rfind('.'))+1:]
            if typeoffile == 'pkl':
                with open(filename, 'rb') as infile:
                    self.model = pickle.load(infile)
                self.log("Model loaded successfully")
                return True
            elif typeoffile == 'pb':
                filename=filename[:(filename.rfind('/'))]
                self.model=tf.contrib.predictor.from_saved_model(filename)
                self.predictor_loaded=True
                self.log("Model loaded successfully")
                return True
            else:
                raise
        except Exception,e:
            self.logger.warning('Failed to load model.')
            self.logger.warning(str(e))
            self.log("Error loading model. Check log for more details.")
            return False
            
    def serving_input_receiver_fn(self):
        inputs = {"x": tf.placeholder(shape=[None, 784], dtype=tf.float32)}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)
    
    def save_model(self, parent,dirname):
        try:
            if self.model is not None:
                if type(self.model)==Pipeline:
                    filename,_ = QtWidgets.QFileDialog.getSaveFileName(
                        parent, 'Save model', dirname, 'Pickle Files (*.pkl)')
                    with open(filename, 'wb') as outfile:
                        pickle.dump(self.model, outfile)
                    self.log("Model saved successfully")
                    return False
                elif type(self.model)==tf.estimator.Estimator:
                    directory = QtWidgets.QFileDialog.getExistingDirectory(
                                parent, 'Select folder',dirname)      
                    self.model.export_savedmodel(directory,self.serving_input_receiver_fn)
                    return False
        except Exception,e:
            self.logger.warning('Failed to save model.')
            self.logger.warning(str(e))
            self.log("Error saving model. Check log for more details.")
            return True
        
############################# MODELS FOR TRAINING #############################
#                                                                             #
# This are models for training / predicting. They follow a pipeline architec- #
# ture of the Sci-kit learn platform, storing the model with its parameters   #
# in the variable self.model. Each one has a detailed description and the     #
# date of creation                                                            #
#                                                                             #
###############################################################################
        
    def RT_and_SVM(self,X,y):
        '''Model that combines Random trees to classificate the images in groups, svd to extrat a set of features, and 
        a SVM classifier. The settings have been optimized for laser cutting on 11-10-2017. Created by Carlos Gonzalez Val'''
        try:
            svc = svm.SVC(C=5000.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                      gamma=0.0005, kernel='rbf', max_iter=-1, probability=False,
                      random_state=None, shrinking=True, tol=0.001, verbose=False)
            t0 = time()
            self.log("Computing projection")
            hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,max_depth=5)
            X_RT = hasher.fit_transform(X)
            pca = decomposition.TruncatedSVD(n_components=20)
            X_transf = pca.fit_transform(X_RT)
            indices = np.random.permutation(len(X_transf))
            self.log("Done. Execution time: " + str(time() - t0))  
            t1 = time()
            self.log("Sorting Data")    
            samples_val=int(len(X_transf)*0.25)
            X_train = X_transf[indices[:-samples_val]]
            y_train = y[indices[:-samples_val]]
            X_test  = X_transf[indices[-samples_val:]]
            y_test  = y[indices[-samples_val:]]
            self.log("Done. Execution time: " + str(time() - t1))
            t1 = time()
            self.log("Training SVM")
            svc.fit(X_train, y_train) 
            self.log("Adjustment score: " + str(svc.score(X_transf,y)))
            self.log("Done. Execution time: " + str(time() - t1))
            self.log("Validating")
            a =svc.predict(X_test)
            self.log("F1 Score: " + str(f1_score(y_test,a)))
            self.log("Done. Total time: " + str(time() - t0))
            self.model=Pipeline(steps=[('reduce_dim1',hasher),
                    ('reduce_dim2',pca),
                    ('svc',svc)])
            return True
        except Exception,e:
            self.logger.warning(str(e))
            self.logger.warning('Error during training of RT+SVM.')
            self.log("Error during training. Check log for more details.")
            return False
    
        
    def PCA_and_SVM(self,X,y):
        '''Model that combines PCA to extrat a set of features, and 
        a SVM classifier. The settings have been optimized for laser cutting on 11-10-2017. Created by Carlos Gonzalez Val'''
        try:
            svc = svm.SVC(C=5000.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                      gamma=0.0005, kernel='rbf', max_iter=-1, probability=False,
                      random_state=None, shrinking=True, tol=0.001, verbose=False)
            t0 = time()
            self.log("Computing PCA projection")
            pca = PCA(n_components=4, 
                  whiten=True).fit(X)
            X_pca = pca.transform(X)
            indices = np.random.permutation(len(X_pca))
            self.log("Done. Execution time: " + str(time() - t0))  
            t1 = time()
            self.log("Sorting Data")   
            samples_val=int(len(X_pca)*0.25)
            X_train = X_pca[indices[:-samples_val]]
            y_train = y[indices[:-samples_val]]
            X_test  = X_pca[indices[-samples_val:]]
            y_test  = y[indices[-samples_val:]]
            self.log("Done. Execution time: " + str(time() - t1))
            t1 = time()
            self.log("Training SVM")
            svc.fit(X_train, y_train) 
            self.log("Adjustment score: " + str(svc.score(X_pca,y)))
            self.log("Done. Execution time: " + str(time() - t1))
            self.log("Validating")
            a =svc.predict(X_test)
            self.log("F1 Score: " + str(f1_score(y_test,a)))
            self.log("Done. Total time: " + str(time() - t0))
            self.model=Pipeline(steps=[('reduce_dim1',pca),
                    ('svc',svc)])
            return True
        except Exception,e:
            self.logger.warning(str(e))
            self.logger.warning('Error during training of PCA+SVM.')
            self.log("Error during training. Check log for more details.")
            return False
            
    def NN_classifier(self,X,y):
        '''Neural network model from fortissimo project. Composed by two 
        convolutional layers with maxpool, a full connected layer with dropout 
        and a full connected layer with softmax'''
        X_FEATURE = 'x'
        try:
            X=(np.array([cv2.resize(x_i,(32,32)) for x_i in X])[:,2:30,2:30])
            indices = np.random.permutation(len(X))
            samples_val=int(len(X)*0.25)
            X_train = X[indices[:-samples_val]]
            y_train = y[indices[:-samples_val]]
            X_test  = X[indices[-samples_val:]]
            y_test  = y[indices[-samples_val:]]
            self.log('Data sorted. Starting training')
            self.max=np.iinfo(X.dtype).max
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={X_FEATURE: X_train.astype(np.float32)/self.max},
                y=y_train.astype(np.int32),
                batch_size=100,
                num_epochs=None,
                shuffle=True)
            test_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={X_FEATURE: X_test.astype(np.float32)/self.max},
                y=y_test.astype(np.int32),
                num_epochs=1,
                shuffle=False)
            ### Convolutional network
            classifier = tf.estimator.Estimator(model_fn=self.classifier_conv_2_layers)
            classifier.train(input_fn=train_input_fn, steps=1000)
            scores = classifier.evaluate(input_fn=test_input_fn)
            self.log('Accuracy of the model: {0:f}'.format(scores['accuracy']))
            self.model= classifier
            return True
        except Exception,e:
            self.logger.warning(str(e))
            self.logger.warning('Error during training of NN.')
            self.log("Error during training. Check log for more details.")
            return False
            
    def classifier_conv_2_layers(self,features, labels, mode):
        '''Neural network model based on fortissimo project. Composed by two 
        convolutional layers with maxpool, a full connected layer with dropout 
        and a full connected layer with softmax'''
        N_CATEGORIES = 2  # Number of categories.
        X_FEATURE = 'x' # Name of the input feature. 
        # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
        # image width and height final dimension being the number of color channels.
        feature = tf.reshape(features[X_FEATURE], [-1, 28, 28, 1])
        # First conv layer will compute 32 features for each 5x5 patch
        with tf.variable_scope('conv_layer1'):
            h_conv1 = tf.layers.conv2d(
                feature,
                filters=32,
                kernel_size=[5, 5],
                padding='same',
                activation=tf.nn.relu)
            h_pool1 = tf.layers.max_pooling2d(h_conv1, pool_size=2, strides=2, padding='same')
        # Second conv layer will compute 64 features for each 5x5 patch.
        with tf.variable_scope('conv_layer2'):
            h_conv2 = tf.layers.conv2d(
                h_pool1,
                filters=64,
                kernel_size=[5, 5],
                padding='same',
                activation=tf.nn.relu)
            h_pool2 = tf.layers.max_pooling2d(h_conv2, pool_size=2, strides=2, padding='same')
            # reshape tensor into a batch of vectors
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        # Densely connected layer with 1024 neurons.
        h_fc1 = tf.layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu)
        if mode == tf.estimator.ModeKeys.TRAIN:
            h_fc1 = tf.layers.dropout(h_fc1, rate=0.5)
        # Compute logits (1 per class) and compute loss.
        logits = tf.layers.dense(h_fc1, N_CATEGORIES, activation=None)
        # Compute predictions.
        predicted_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'class': predicted_classes,'prob': tf.nn.softmax(logits)}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions,export_outputs={'predictions':tf.estimator.export.PredictOutput({'y':predicted_classes})})
        # Compute loss.
        onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), N_CATEGORIES, 1, 0)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
        # Create training op.
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        # Compute evaluation metrics.
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(
                           labels=labels, predictions=predicted_classes)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
        
    def regressor_conv_2_layers(self,features, labels, mode):
        '''Neural network model from fortissimo project. Composed by two 
        convolutional layers with maxpool, a full connected layer with dropout 
        and a full connected layer for regression '''
        N_CATEGORIES = 2  # Number of features to predict (lenght of labels vector)
        X_FEATURE = 'x' # Name of the input feature. 
        # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
        # image width and height final dimension being the number of color channels.
        feature = tf.reshape(features[X_FEATURE], [-1, 28, 28, 1])
        # First conv layer will compute 32 features for each 5x5 patch
        with tf.variable_scope('conv_layer1'):
            h_conv1 = tf.layers.conv2d(
                feature,
                filters=32,
                kernel_size=[5, 5],
                padding='same',
                activation=tf.nn.relu)
            h_pool1 = tf.layers.max_pooling2d(h_conv1, pool_size=2, strides=2, padding='same')
        # Second conv layer will compute 64 features for each 5x5 patch.
        with tf.variable_scope('conv_layer2'):
            h_conv2 = tf.layers.conv2d(
                h_pool1,
                filters=64,
                kernel_size=[5, 5],
                padding='same',
                activation=tf.nn.relu)
            h_pool2 = tf.layers.max_pooling2d(h_conv2, pool_size=2, strides=2, padding='same')
            # reshape tensor into a batch of vectors
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        # Densely connected layer with 1024 neurons.
        h_fc1 = tf.layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu)
        if mode == tf.estimator.ModeKeys.TRAIN:
            h_fc1 = tf.layers.dropout(h_fc1, rate=0.5)
        # Compute regression (1 per parameter) and compute loss.
        output_layer = tf.layers.dense(h_fc1, N_CATEGORIES, activation=None)
         # Reshape output layer to 1-dim Tensor to return predictions
        predictions = tf.reshape(output_layer, [-1])

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'value': predictions}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions,export_outputs={'predictions':tf.estimator.export.PredictOutput({'y':predictions})})
        # Compute loss.
        loss = tf.losses.mean_squared_error(labels, predictions)
        # Create training op.
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        # Compute evaluation metrics.
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(
                           labels=labels, predictions=predictions)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
        
################## VISUALIZATION AND DIMENSIONALITY REDUCTION #################
#                                                                             #
# This are models for visualicing high dimensional data. They perform a       #
# dimensionality reduction to 2 or 3 features and plots the results so the    #
# datasets can be more easily understood                                      #
#                                                                             #
###############################################################################

    # PCA projection of the digits dataset
    def PCA_proj(self,X,y):
        self.log("Computing PCA projection with "+ str(len(X))+ " samples")
        t0 = time()
        X_pca = decomposition.TruncatedSVD(n_components=3).fit_transform(X)
        self.log("Done. Execution time: "+ str(time() - t0))
        self.plot_features_3d(X_pca,y,
                       "Principal Components projection of the digits (time %.2fs)" %
                       (time() - t0))
                       
    # Random Trees projection of the digits dataset                   
    def RT_proj(self,X,y):
        self.log("Computing Totally Random Trees embedding with "+ str(len(X))+ " samples")
        t0 = time()
        hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                               max_depth=5)
        X_transformed = hasher.fit_transform(X)
        pca = decomposition.TruncatedSVD(n_components=3)
        X_reduced = pca.fit_transform(X_transformed)
        self.log("Done. Execution time: "+ str(time() - t0))
        self.plot_features_3d(X_reduced,y,
                       "Random forest embedding of the digits (time %.2fs)" %
                       (time() - t0))

    # Projection on to the first 2 linear discriminant components
    def LDA_proj(self,X,y):
        self.log("Computing Linear Discriminant Analysis projection with "+ str(len(X))+ " samples")
        t0 = time()
        X2 = X.astype(float)
        X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
        X_lda = lda.LDA(n_components=2).fit_transform(X2, y)
        self.plot_features_2d(X_lda,y,
                       "Linear Discriminant projection of the digits (time %.2fs)" %
                       (time() - t0))
    
    # Isomap projection of the digits dataset
    def isomap_proj(self,X,y):
        self.log("Computing Isomap embedding with "+ str(len(X))+ " samples")
        t0 = time()
        X_iso = manifold.Isomap(self.n_neighbors, n_components=3).fit_transform(X)
        self.log("Done. Execution time: "+ str(time() - t0))
        self.plot_features_3d(X_iso,y,
                       "Isomap projection of the digits (time %.2fs)" %
                       (time() - t0))
    
    # Locally linear embedding of the digits dataset
    def LLE_proj(self,X,y):
        self.log("Computing LLE embedding with "+ str(len(X))+ " samples")
        t0 = time()
        X3 = X.astype(float)
        clf = manifold.LocallyLinearEmbedding(self.n_neighbors, n_components=3,
                                              method='standard')
        X_lle = clf.fit_transform(X3)
        self.log("Done. Execution time: "+ str(time() - t0) + ". Reconstruction error: " + str(clf.reconstruction_error_))
        self.plot_features_3d(X_lle,y,
                       "Locally Linear Embedding of the digits (time %.2fs)" %
                       (time() - t0))
    
    # Modified Locally linear embedding of the digits dataset
    def MLLE_proj(self,X,y):
        self.log("Computing modified LLE embedding with "+ str(len(X))+ " samples")
        t0 = time()
        clf = manifold.LocallyLinearEmbedding(self.n_neighbors, n_components=3,
                                              method='modified')
        X_mlle = clf.fit_transform(X.astype(float))
        self.log("Done. Execution time: "+ str(time() - t0) + ". Reconstruction error: " + str(clf.reconstruction_error_))
        self.plot_features_3d(X_mlle,y,
                       "Modified Locally Linear Embedding of the digits (time %.2fs)" %
                       (time() - t0))
    
    # HLLE embedding of the digits dataset
    def HLLE_proj(self,X,y):
        self.log("Computing Hessian LLE embedding with "+ str(len(X))+ " samples")
        t0 = time()
        clf = manifold.LocallyLinearEmbedding(self.n_neighbors, n_components=3,
                                              method='hessian')
        X_hlle = clf.fit_transform(X.astype(float))
        self.log("Done. Execution time: "+ str(time() - t0) + ". Reconstruction error: " + str(clf.reconstruction_error_))
        self.plot_features_3d(X_hlle,y,
                       "Hessian Locally Linear Embedding of the digits (time %.2fs)" %
                       (time() - t0))
    
    # LTSA embedding of the digits dataset
    def LTSA_proj(self,X,y):
        self.log("Computing LTSA embedding with "+ str(len(X))+ " samples")
        t0 = time()
        clf = manifold.LocallyLinearEmbedding(self.n_neighbors, n_components=3,
                                              method='ltsa')
        X_ltsa = clf.fit_transform(X.astype(float))
        self.log("Done. Execution time: "+ str(time() - t0) + ". Reconstruction error: " + str(clf.reconstruction_error_))
        self.plot_features_3d(X_ltsa,y,
                       "Local Tangent Space Alignment of the digits (time %.2fs)" %
                       (time() - t0))
    
    # MDS  embedding of the digits dataset
    def MDS_proj(self,X,y):
        self.log("Computing MDS embedding with "+ str(len(X))+ " samples")
        t0 = time()
        clf = manifold.MDS(n_components=3, n_init=1, max_iter=100)
        X_mds = clf.fit_transform(X.astype(float))
        self.log("Done. Execution time: "+ str(time() - t0) + ". Stress: " + str(clf.stress_))
        self.plot_features_3d(X_mds,y,
                       "MDS embedding of the digits (time %.2fs)" %
                       (time() - t0))

    # Spectral embedding of the digits dataset
    def CSE_proj(self,X,y):
        self.log("Computing Spectral embedding with "+ str(len(X))+ " samples")
        t0 = time()
        embedder = manifold.SpectralEmbedding(n_components=3, random_state=0,
                                              eigen_solver="arpack")
        X_se = embedder.fit_transform(X)
        self.log("Done. Execution time: "+ str(time() - t0))
        self.plot_features_3d(X_se,y,
                       "Spectral embedding of the digits (time %.2fs)" %
                       (time() - t0))
    
    # t-SNE embedding of the digits dataset
    def tsne_proj(self,X,y):
        self.log("Computing t-SNE embedding with "+ str(len(X))+ " samples")
        t0 = time()
        tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X)
        self.log("Done. Execution time: "+ str(time() - t0))
        self.plot_features_3d(X_tsne,y,
                       "t-SNE embedding of the digits (time %.2fs)" %
                       (time() - t0))       
                       
    # Plot features in a 2d figure                   
    def plot_features_2d(self,features, targets,title, idx=[0, 1]):
        self.logger.info('Plotting 2d features.')
        labels = np.unique(targets)
        colors = 'bgrcmykbgrcmykbgrcmykbgrcmyk'
        plt.figure()
        for k, label in enumerate(labels):
            data = features[targets == label]
            col = colors[k]
            plt.plot(data[:, idx[0]], data[:, idx[1]],
                     col + '.', label=str(label))
        plt.legend()
        plt.show()
    
    # Plot features in a 3d figure
    def plot_features_3d(self,features, targets,title, idx=[0, 1, 2]):
        self.logger.info('Plotting 3d features.')
        if len(features) > 5000:
            features, targets = shuffle(
                features, targets, n_samples=5000, random_state=0)
        labels = np.unique(targets)
        colors = 'bgrcmykbgrcmykbgrcmykbgrcmyk'
        fig = plt.figure(figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.set_title(title)
        rects, rlabels = [], []
        for k, label in enumerate(labels):
            data = features[targets == label]
            label = '%s - %i' % (label, len(data))
            ax.scatter(data[:, idx[0]], data[:, idx[1]], data[:, idx[2]],
                       c=colors[k], label=label)
            rects.append(plt.Rectangle((0, 0), 1, 1, fc=colors[k]))
            rlabels.append(label)
        ax.legend(rects, rlabels, fontsize=11)
        plt.show()