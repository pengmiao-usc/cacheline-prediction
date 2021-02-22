# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:33:30 2020

@author: pmzha
"""

import os
import numpy as np
#from Evaluate import Evaluate

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import LSTM
from keras.layers import CuDNNLSTM  #use env:ef1 to train with GPU
from tensorflow.keras.callbacks import EarlyStopping
from tcn import TCN, tcn_full_summary

class tcn_model:
    def __init__(self,
                 vocab_size,
                 batch_size,
                 embedding_dim,
                 i_dim,
                 o_dim,
                 num_filters=200):
        self.i_dim = i_dim
        self.o_dim = o_dim

        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embedding_dim,
                                 input_length=i_dim))
        self.model.add(TCN(nb_filters = num_filters, kernel_size=2, dilations=[1,2,4,8]))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(o_dim, activation='sigmoid'))
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        self.model_name = 'TCN'
        tcn_full_summary(self.model, expand_residual_blocks=True)

    def train(self, X_train, y_train, X_test, y_test, num_epochs, batch_size):
        history = self.model.fit(X_train,
                                 y_train,
                                 epochs=num_epochs,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 validation_data=(X_test, y_test))
        return history

    def predict(self, X):
        return self.model.predict(np.asarray(X))


class tcn_model_stateful:
    def __init__(self, vocab_size, batch_size, embedding_dim, i_dim, o_dim):
        self.i_dim = i_dim
        self.o_dim = o_dim

        self.model = Sequential()
        self.model.add(
            Embedding(vocab_size,
                      embedding_dim,
                      input_length=i_dim,
                      batch_input_shape=(batch_size, i_dim)))

        self.model.add(TCN(nb_filters = 64, kernel_size=3, dilations=[1,2,4,8,16,32,64]))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(20, activation='sigmoid'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(o_dim, activation='sigmoid'))
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

        self.model_name = 'TCN'
        self.model.summary()
        tcn_full_summary(self.model, expand_residual_blocks=False)

    def train(self, X_train, y_train, X_test, y_test, num_epochs, batch_size):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=3,
                                       verbose=1)
        history = self.model.fit(X_train,
                                 y_train,
                                 epochs=num_epochs,
                                 shuffle=False,
                                 batch_size=batch_size,
                                 validation_data=(X_test, y_test),
                                 callbacks=[early_stopping])
        return history

    def predict(self, X, batch_size):
        return self.model.predict(np.asarray(X), batch_size=batch_size)
