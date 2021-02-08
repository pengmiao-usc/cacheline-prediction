# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 23:56:18 2020

@author: pmzha
"""

import csv
import os
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer

def convert_to_dt(source_path):
    dataset = pd.read_csv(source_path, header=None, index_col=None)

    '''conver to deltas'''
    dataset_dt=dataset.diff()[1:]
    dataset_dt.to_csv("dt.csv", index=None)
    
def create_windowed_dataset(data, look_back):
    """
    Create the dataset by grouping windows of memory accesses together (using the look_back parameter)

    data: it should be a list of integers
    """
    sequences = list()
    for i in range(look_back, len(data)):
        sequence = data[i - look_back:i + 1]
        sequences.append(sequence)
    return np.array(sequences)

def difference(dataset, interval=1):
    """
    Calculates the difference between a time-series and a lagged version of it
    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

def convert_to_binary(data,bit_size=16):
    s='{0:0'+str(bit_size)+'b}'
    data_binary = np.array([[int(d) for i in j for d in str(s.format(i))] for j in list(data)])
    return data_binary    

def convert_binary_to_dec(data):
    #return np.reshape(np.array(list([int("".join(str(i) for i in List),2) for List in data])),(-1,1))
    #dec = np.packbits(np.array(data, dtype=np.bool).reshape(-1, bit_size//8, 8)[:, ::-1]).view(np.uint16)
    dec = np.reshape(np.array(list([int("".join(str(int(i)) for i in List),2) for List in data])),(-1,1))
            
    return dec
    # return data_dec 

def token_back(y_pred_dec,y_test_dec,tokenizer)  :
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
        words = [reverse_word_map.get(letter[0]) for letter in list_of_indices]
        return(words)
    dummy_word = "0xffffffff"    
    original_testing_diffs = list(map(sequence_to_text, [y_pred_dec]))
    original_predictions_diffs = list(map(sequence_to_text, [y_test_dec]))
    tmp = [((-1 if int(k[0]) == 1 else 1)*int(k[2:]), (-1 if int(l[0]) == 1 else 1)*int(l[2:])) if l is not None and k is not None and l != dummy_word and k != dummy_word else (None, None) for k,l in zip(original_testing_diffs[0], original_predictions_diffs[0])]
    original_testing_diffs, original_predictions_diffs = zip(*tmp)   
    return list(original_testing_diffs), list(original_predictions_diffs)

    
    

