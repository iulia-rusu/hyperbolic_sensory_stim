"Processing pipeline for MNE"

import os
import tensorflow as tf
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Activation
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import chain



directrory = os.getcwd()



#get working directory
#set working directory to 

"load data from ProcAiry using absolute path. Stimulus has two versions: 0 and 1, prepare both arrays"
def stim(x):
    s = np.load('/Users/iuliarusu/Documents/Sharpee/ProcAiryData/inpfull_op50_SF.npz')
    s_concat = sum(s.values(, []))

    for key in s.keys():
        stimulus = np.concatenate(s[key], axis = 1)
        stimulus = stimulus.T
    return stimulus

"curent arrangement of stim type and worm #, stim0 -> w0, w1, w2, stim1 -> w3. w4. w5. w6"
    
            
            
            
    s_con = np.concatenate([s['arr_0'], s['arr_1'], s['arr_2'], s['arr_3'], s['arr_4'], s['arr_5'], s['arr_6']], axis =1)
    s_con = s_con.T
