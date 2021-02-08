#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 18:08:09 2021

@author: pengmiao
"""

import csv
import os
import pandas as pd
import numpy as np
from numpy import argmax
import pickle
import pdb
import functions as f
import glob
import os
from tqdm import tqdm
from statistics import mean 
import random
from os import path
from sklearn.metrics import roc_curve,f1_score,recall_score,precision_score,accuracy_score
#pdb.set_trace()
#%% FUNCTION: UNIFORM ACTIVATION
'''
Uniform Activation baseline

Average number of cache line activated m in M
During prediction, select each label with probability m/64:
    mean_label_num:mean number of activated cache lines in bitmap vector (label)
    gen_list: using probability = 1-mean_label_num/64 to select 0; 
                    probability =mean_label_num/64 to select 1;
    directly generate bitmap, evaluate with labels.
'''

def baseline_uniformly_prefetch(file_path):
    df_x = pd.read_pickle(file_path)
    split2=len(df_x)*3//4
    df_eva=df_x[split2:]
    y_test=np.stack(df_eva['y'])
    
    length=[len(x) for x in df_x['label']]
    mean_label_num=mean(length)
    #gen_list=np.random.choice([0,1], 64, p=[1-mean_label_num/64,mean_label_num/64])
    
    y_pred_bin=[]
    for i in range(len(y_test)):
        y_pred_bin.append(np.random.choice([0,1], 64, p=[1-mean_label_num/64,mean_label_num/64]))
        
    f1_score_res=f1_score(y_test, y_pred_bin, average='samples')
    #recall: tp / (tp + fn)
    recall_score_res=recall_score(y_test, y_pred_bin, average='samples')
    #precision: tp / (tp + fp)
    precision_score_res=precision_score(y_test, y_pred_bin, average='samples',zero_division=0)
    return [f1_score_res,recall_score_res,precision_score_res]

#%%
M,N=500,64
LEN=400000

DATASET_READ_DIR = "./dataset_pickle/dataset1_M_"+str(M)+"_N_"+str(N)+"/XY/"
BASELINE_SAVE_DIR = "./dataset_pickle/dataset1_M_"+str(M)+"_N_"+str(N)+"/baseline/"
if not os.path.exists(BASELINE_SAVE_DIR):
    os.makedirs(BASELINE_SAVE_DIR)
file_list = sorted(glob.glob(DATASET_READ_DIR+"*"))

#%%
df_res=pd.DataFrame([])

for file in file_list:
    res=baseline_uniformly_prefetch(file)
    file_name = file.split("XY/")[-1].split("_xy")[-2]
    df_res[file_name]=res
    
#np_save_path=BASELINE_SAVE_DIR+"uniform.pk"
#df_res.to_pickle(np_save_path)


#%% BASELINE 2 Biased 













import csv
import os
import pandas as pd
import numpy as np
from numpy import argmax
import pickle
import pdb
import glob
import os
from tqdm import tqdm
from statistics import mean 
import random
from os import path
from sklearn.metrics import roc_curve,f1_score,recall_score,precision_score,accuracy_score


#%% FUNCTION: BIASED ACTIVATION
from collections import Counter
from collections import OrderedDict
def baseline_biased_prefetch(file_path):
    df_x = pd.read_pickle(file_path)
    split2=len(df_x)*3//4
    df_eva=df_x[split2:]
    y_test=np.stack(df_eva['y'])
    len_df_eva=len(df_eva)
    total=len(df_x)
    
    #length=[len(x) for x in df_x['label']]
    label_list=list(df_x['label'].values)
    
    count = Counter([])
    for label in label_list:
        count += Counter(label)
    
    for key in count.keys():
        count[key]=count[key]/total
    
    count_order=OrderedDict(sorted(count.items()))
  #  prob=list(count_order.values())
    
    y_pred_bin_ls=[]
    for i in tqdm(range(64)):
        if i in count_order:
            random_arr=np.random.choice([0,1],len_df_eva, p=[1-count_order[i],count_order[i]])
        else:
            random_arr=np.zeros(len_df_eva)
        #y_pred_bin=np.append([y_pred_bin],[random_arr],axis=0)
        y_pred_bin_ls.append(random_arr)
    
    y_pred_bin=np.transpose(np.array(y_pred_bin_ls))
      
    f1_score_res=f1_score(y_test, y_pred_bin, average='samples')
    #recall: tp / (tp + fn)
    recall_score_res=recall_score(y_test, y_pred_bin, average='samples')
    #precision: tp / (tp + fp)
    precision_score_res=precision_score(y_test, y_pred_bin, average='samples',zero_division=0)


    return [f1_score_res,recall_score_res,precision_score_res]      
#%%  
M,N=500,64
LEN=400000

DATASET_READ_DIR = "./dataset_pickle/dataset1_M_"+str(M)+"_N_"+str(N)+"/XY/"
BASELINE_SAVE_DIR = "./dataset_pickle/dataset1_M_"+str(M)+"_N_"+str(N)+"/baseline/"
if not os.path.exists(BASELINE_SAVE_DIR):
    os.makedirs(BASELINE_SAVE_DIR)
file_list = sorted(glob.glob(DATASET_READ_DIR+"*"))
file_path=file_list[0]

res=baseline_biased_prefetch(file_list[0])
print(res)

#%%
#pdb.set_trace()
file='./dataset_pickle/dataset1_M_500_N_64/XY/621.wrf_xy.pk'
res=baseline_biased_prefetch(file)




















