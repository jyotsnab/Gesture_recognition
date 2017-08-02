# -*- coding: utf-8 -*-
"""
main_model.py
@author: jyotsna
"""

#imports
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import Build_data_set as BDS
from keras.utils.np_utils import to_categorical
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras import optimizers


def train_model(folder_path):
    """
    Parameters
    ----------
    folder_path : string
        path to the folder containing individual 'txt' sample files
    
    Returns
    -------
    scaler : Scaling function fitted to train set
    model : learning model fitted to training set
    
    Remarks
    -------
    The function follows the order of execution:
        1) Build dataset by calling feature engineering functions
        2) Split dataset into train and test sets
        3) Scale train set
        4) Transform test set based on function in 3)
        5) One-hot encoding of y_train
        6) Fit multi layer neural network model on train set
        7) return scale function (3) and model funtion (6)
        
    
    """
    #input folder path to the  folder containing all training samples to build features
    #folder_path= r'..\Gesture_Data'
    df = BDS.build_training_set(folder_path)
    cols= df.columns
    
    # Splitting dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(df[cols[:-1]], df[cols[-1]], test_size=0.15, random_state=0)
    
    # Standard Scaling using train set only
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Convert Interger-encoded categorical variables as one-hot encoded values
    Y_train = to_categorical(y_train)
    
    
    # Building Neural Network Layers
    N = X_train.shape[1]
    H = 100
    K = 7
    model = Sequential()
    model.add(Dense(H, input_dim=N))
    model.add(Activation("relu"))
    model.add(Dense(H))
    model.add(Activation("relu"))
    model.add(Dense(K))
    model.add(Activation("softmax"))
    model.compile(optimizer="adam", loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, nb_epoch=15, batch_size=32)
    
    print ('accuracy on X_test: ')
    y_predicted = model.predict_classes(X_test, verbose=0)
    print("accuracy on test set: %0.3f" % np.mean(y_predicted == y_test))
    
    return scaler,model

def predict_sample(sample_file_path, scaler, model):
    """
    Parameters
    ----------
    sample_file_path : string
        path to 'txt' file of the new sample containing 50*8 values
    scaler : function
        preprocessing function from 'fit' on training set
    model : function
        learning model from 'fit' on training set
    
    Returns
    -------
    y_pred : int
        predicted gesture class
   
    
    Remarks
    -------
    The function follows the order of execution:
        1) Load sample array from 'txt' file
        2) Build features 
        3) Scale transform based on 'scaler' funtion 
        4) perform prediction using 'model'
        
    """
    # single sample of shape 50*8 only
    
    #load sample as array
    sample = np.loadtxt(sample_file_path, delimiter=',')
    #Build features
    X_test= BDS.transform_sample(sample)    
    X_test= X_test.reshape(1,-1)
    X_test = scaler.transform(X_test)
    
    #predict from model
    y_pred = model.predict_classes(X_test, verbose=0)
    
    return y_pred
    

class Multilayer_NN(object):
    """
    Creates an instance of the model.
    
    """
    
    def __init__(self):
        """
        intiates the model
        return: None
        """
        
        
    
    def fit(self, folder_path):
        """
        Fits the training samples to a model.
        Holds the parameters of preprocessing and learning models.
        
        """
        
        scaler,model = train_model(folder_path)
        self.model = model
        self.scaler = scaler
        
        return self
    
    def predict(self, sample_file_path):
        """
        Prints out the gesture class prediction for the specific instance 
        of the model.
        """
        
        y_pred = predict_sample(sample_file_path, self.scaler,self.model)
        
        print ('Predicted Gesture Class: ', y_pred)
        
        return y_pred


""" Implementation """

ML_NN = Multilayer_NN()

ML_NN.fit(r'..\Gesture_Data')
ML_NN.predict(r'..\Gesture_Data\new_sample.txt')

    
    
        