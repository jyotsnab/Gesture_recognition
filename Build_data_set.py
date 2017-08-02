# -*- coding: utf-8 -*-
"""
Build_data_set.py
@author: jyotsna
"""

import numpy as np
import pandas as pd
from glob import glob
import Feature_Engineering as fe

#folder_path = r'..\ThalmicLabs\Gesture_Data'
def build_training_set(folder_path):
    """
    Build training set from given samples:
        
    Parameters
    ----------
    folder_path: string
        path to the  folder containing samples
        Ex: r'..\ThalmicLabs\Gesture_Data'

    Returns
    -------
    df : pandas dataframe
        dataframe containing features and target
        this dataframe is also saved in the same folder path as 'gesture_dataframe.csv'
        
    """
    files ={}
    for i in range(1,7):
        files[i]= glob(folder_path+ '/Gesture'+str(i)+'_*.txt')
        
    columns =['stddev_1','variance_1', 'mean_1','peak_1','area_1','hist_1_1','hist_1_2','hist_1_3','hist_1_4',
              'stddev_2','variance_2', 'mean_2','peak_2','area_2','hist_2_1','hist_2_2','hist_2_3','hist_2_4',
              'stddev_3','variance_3', 'mean_3','peak_3','area_3','hist_3_1','hist_3_2','hist_3_3','hist_3_4',
              'stddev_4','variance_4', 'mean_4','peak_4','area_4','hist_4_1','hist_4_2','hist_4_3','hist_4_4',
              'stddev_5','variance_5', 'mean_5','peak_5','area_5','hist_5_1','hist_5_2','hist_5_3','hist_5_4',
              'stddev_6','variance_6', 'mean_6','peak_6','area_6','hist_6_1','hist_6_2','hist_6_3','hist_6_4',
              'stddev_7','variance_7', 'mean_7','peak_7','area_7','hist_7_1','hist_7_2','hist_7_3','hist_7_4',
              'stddev_8','variance_8', 'mean_8','peak_8','area_8','hist_8_1','hist_8_2','hist_8_3','hist_8_4',
              'gesture'
              ]
    index_range = 2000*6
    df = pd.DataFrame([],columns=columns,index=range(index_range))
    index=0
    for i in files.keys():
        for sample in range(len(files[i])):
            #loads each sample as a numpy array of shape 50*8
            data = np.loadtxt(files[i][sample],delimiter=',')
            #normalize data
            data = fe.normalize(data)
            # create features
            df.loc[index][columns[:5]] = fe.stats(data[:,0])
            df.loc[index][columns[9:14]] = fe.stats(data[:,1])
            df.loc[index][columns[18:23]] = fe.stats(data[:,2])
            df.loc[index][columns[27:32]] = fe.stats(data[:,3])
            df.loc[index][columns[36:41]] = fe.stats(data[:,4])
            df.loc[index][columns[45:50]] = fe.stats(data[:,5])
            df.loc[index][columns[54:59]] = fe.stats(data[:,6])
            df.loc[index][columns[63:68]] = fe.stats(data[:,7])
            df.loc[index][columns[5:9]] = fe.hist(data[:,0])
            df.loc[index][columns[14:18]] = fe.hist(data[:,1])
            df.loc[index][columns[23:27]] = fe.hist(data[:,2])
            df.loc[index][columns[32:36]] = fe.hist(data[:,3])
            df.loc[index][columns[41:45]] = fe.hist(data[:,4])
            df.loc[index][columns[50:54]] = fe.hist(data[:,5])
            df.loc[index][columns[59:63]] = fe.hist(data[:,6])
            df.loc[index][columns[68:72]] = fe.hist(data[:,7])
            df.loc[index]['gesture'] = i
            index += 1
            
    # saving training dataset
    df.to_csv(folder_path + '\gesture_dataframe.csv')
    return df
    
    
def transform_sample(data):
    
    """
    Build features for a new sample:
        
    Parameters
    ----------
    data: path to the  folder containing samples
        Ex: r'..\ThalmicLabs\Gesture_Data'

    Returns
    -------
    test_sample : vector
        vector containing features derived feature engineering of the 
        sample data
        
    """
    data = fe.normalize(data)
    test_sample = np.zeros(72)
    test_sample[:5] = fe.stats(data[:,0])
    test_sample[9:14] = fe.stats(data[:,1])
    test_sample[18:23] = fe.stats(data[:,2])
    test_sample[27:32] = fe.stats(data[:,3])
    test_sample[36:41] = fe.stats(data[:,4])
    test_sample[45:50] = fe.stats(data[:,5])
    test_sample[54:59] = fe.stats(data[:,6])
    test_sample[63:68] = fe.stats(data[:,7])
    test_sample[5:9] = fe.hist(data[:,0])
    test_sample[14:18] = fe.hist(data[:,1])
    test_sample[23:27] = fe.hist(data[:,2])
    test_sample[32:36] = fe.hist(data[:,3])
    test_sample[41:45] = fe.hist(data[:,4])
    test_sample[50:54] = fe.hist(data[:,5])
    test_sample[59:63] = fe.hist(data[:,6])
    test_sample[68:72] = fe.hist(data[:,7])
    
    return test_sample
        
        