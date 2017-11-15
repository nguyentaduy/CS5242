
# coding: utf-8

# In[185]:

from __future__ import print_function
import json
from pprint import pprint
import numpy as np

import keras
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from keras import optimizers
from keras import layers
from keras.layers import Lambda, Activation,recurrent, Bidirectional, Dense, Flatten, Conv1D, Dropout, LSTM, GRU, concatenate, multiply, add, Reshape, MaxPooling1D, BatchNormalization
from keras.models import Model, load_model
from keras import backend as K
from keras.engine.topology import Layer

from preprocess import *


class MyLayer(Layer):

    def __init__(self, output_dim, name, **kwargs):
        self.output_dim = output_dim
        self.name = name
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name=self.name,
                                      shape=(input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.W)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.output_dim)

def DrQA(model_weights=None):

    P = layers.Input(shape=(max_para,dimension), dtype='float32')
    Q = layers.Input(shape=(max_q,dimension), dtype='float32')
    P_exact_match = layers.Input(shape=(max_para,3), dtype='float32')

    alpha = MyLayer(128,"alpha")
    beta = MyLayer(1,"beta")
    gamma_s = MyLayer(128, "gamma_s")
    gamma_e = MyLayer(128, "gamma_e")
    hidden_unit = 64

    aligned_P = layers.Activation("relu")(alpha(P))
    aligned_Q = layers.Activation("relu")(alpha(Q))
    coatt_aligned_PQ = layers.dot([aligned_Q, aligned_P], axes=2)
    coatt_aligned_PQ = Reshape((max_para * max_q,))(coatt_aligned_PQ)
    coatt_aligned_PQ = layers.Activation("softmax")(coatt_aligned_PQ)
    coatt_aligned_PQ = Reshape((max_q, max_para,))(coatt_aligned_PQ)
    coatt_aligned_PQ = layers.dot([coatt_aligned_PQ,Q], axes=1)

    final_P = layers.concatenate([P, P_exact_match, coatt_aligned_PQ], axis=2)
    final_P = Dropout(.3)(final_P)
    final_P = Bidirectional(LSTM(hidden_unit,return_sequences=True))(final_P)
    final_P = Dropout(.3)(final_P)
    final_P = Bidirectional(LSTM(hidden_unit,return_sequences=True))(final_P)
    final_P = Dropout(.3)(final_P)
    final_P = Bidirectional(LSTM(hidden_unit,return_sequences=True))(final_P)
    final_P = Dropout(.3)(final_P)

    final_Q = Dropout(.3)(Q)
    final_Q = Bidirectional(LSTM(hidden_unit,return_sequences=True))(final_Q)
    final_Q = Dropout(.3)(final_Q)
    final_Q = Bidirectional(LSTM(hidden_unit,return_sequences=True))(final_Q)
    final_Q = Dropout(.3)(final_Q)
    final_Q = Bidirectional(LSTM(hidden_unit,return_sequences=True))(final_Q)
    final_Q = Dropout(.3)(final_Q)
    w = beta(final_Q)
    final_Q = layers.dot([w,final_Q],axes=1)

    start = gamma_s(final_P)
    start = layers.dot([start, final_Q], axes=2)
    start = Reshape((max_para,))(start)
    start = layers.Activation("softmax")(start)

    end = gamma_e(final_P)
    end = layers.dot([end, final_Q], axes=2)
    end = Reshape((max_para,))(end)
    end = layers.Activation("softmax")(end)

    model = Model([P, Q, P_exact_match],[start, end])
    if model_weights != None:
        model.load_weights(model_weights)


    model.compile(optimizer="adamax",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def Fusion(model_weights=None):

    P = layers.Input(shape=(max_para,dimension), dtype='float32')
    Q = layers.Input(shape=(max_q,dimension), dtype='float32')
    P_exact_match = layers.Input(shape=(max_para,3), dtype='float32')

    alpha_w = MyLayer(128,"alpha_w")
    alpha_l = MyLayer(128,"alpha_l")
    alpha_h = MyLayer(128,"alpha_h")
    alpha_u = MyLayer(128,"alpha_u")
    alpha_s = MyLayer(128,"alpha_s")
    beta = MyLayer(1,"beta")
    gamma_s = MyLayer(250, "gamma_s")
    gamma_e = MyLayer(250, "gamma_e")
    hidden_unit = 125

    # Word level fusion
    aligned_P = layers.Activation("relu")(alpha_w(P))
    aligned_Q = layers.Activation("relu")(alpha_w(Q))
    word_fusion = layers.dot([aligned_Q, aligned_P], axes=2)
    word_fusion = Reshape((max_para * max_q,))(word_fusion)
    word_fusion = layers.Activation("softmax")(word_fusion)
    word_fusion = Reshape((max_q, max_para,))(word_fusion)
    word_fusion = layers.dot([word_fusion,Q],axes=1)

    # READING
    final_P = layers.concatenate([P, P_exact_match, word_fusion], axis=2)
    final_P = Dropout(.4)(final_P)
    low_P = Bidirectional(LSTM(hidden_unit,return_sequences=True))(final_P)
    low_P = Dropout(.4)(low_P)
    high_P = Bidirectional(LSTM(hidden_unit,return_sequences=True))(low_P)
    high_P = Dropout(.4)(high_P)

    final_Q = Dropout(.4)(Q)
    low_Q = Bidirectional(LSTM(hidden_unit,return_sequences=True))(final_Q)
    low_Q = Dropout(.4)(low_Q)
    high_Q = Bidirectional(LSTM(hidden_unit,return_sequences=True))(low_Q)
    high_Q = Dropout(.4)(high_Q)

    # Question understanding
    U_Q = layers.concatenate([low_Q, high_Q], axis=2)
    U_Q = Bidirectional(LSTM(hidden_unit,return_sequences=True))(U_Q)
    U_Q = Dropout(.4)(U_Q)
    w = beta(U_Q)
    UQ = layers.dot([w,U_Q],axes=1)

    # History of words
    HP = layers.concatenate([P, low_P, high_P], axis = 2)
    HQ = layers.concatenate([Q, low_Q, high_Q], axis = 2)

    # Low level fusion
    low_HP = layers.Activation("relu")(alpha_l(HP))
    low_HQ = layers.Activation("relu")(alpha_l(HQ))
    low_fusion = layers.dot([low_HQ, low_HP], axes=2)
    low_fusion = Reshape((max_para * max_q,))(low_fusion)
    low_fusion = layers.Activation("softmax")(low_fusion)
    low_fusion = Reshape((max_q, max_para,))(low_fusion)
    low_fusion = layers.dot([low_fusion,low_Q],axes=1)
    low_fusion = Dropout(.4)(low_fusion)

    # High level fusion
    high_HP = layers.Activation("relu")(alpha_h(HP))
    high_HQ = layers.Activation("relu")(alpha_h(HQ))
    high_fusion = layers.dot([high_HQ, high_HP], axes=2)
    high_fusion = Reshape((max_para * max_q,))(high_fusion)
    high_fusion = layers.Activation("softmax")(high_fusion)
    high_fusion = Reshape((max_q, max_para,))(high_fusion)
    high_fusion = layers.dot([high_fusion,high_Q],axes=1)
    high_fusion = Dropout(.4)(high_fusion)

    # Understanding level fusion
    U_HP = layers.Activation("relu")(alpha_u(HP))
    U_HQ = layers.Activation("relu")(alpha_u(HQ))
    U_fusion = layers.dot([U_HQ, U_HP], axes=2)
    U_fusion = Reshape((max_para * max_q,))(U_fusion)
    U_fusion = layers.Activation("softmax")(U_fusion)
    U_fusion = Reshape((max_q, max_para,))(U_fusion)
    U_fusion = layers.dot([U_fusion,U_Q],axes=1)
    U_fusion = Dropout(.4)(U_fusion)

    # Fully-aware attention
    P_fusion = layers.concatenate([low_P, high_P, low_fusion, high_fusion, U_fusion], axis=2)
    V_P = Bidirectional(LSTM(hidden_unit,return_sequences=True))(P_fusion)
    V_P = Dropout(.4)(V_P)
    H_P = layers.concatenate([P, P_fusion, V_P], axis=2)

    # Self-boosted fusion
    self_HP = layers.Activation("relu")(alpha_s(H_P))
    self_fusion = layers.dot([self_HP, self_HP], axes=2)
    self_fusion = Reshape((max_para * max_para,))(self_fusion)
    self_fusion = layers.Activation("softmax")(self_fusion)
    self_fusion = Reshape((max_para, max_para))(self_fusion)
    self_fusion = layers.dot([self_fusion,V_P],axes=1)
    self_fusion = Dropout(.4)(self_fusion)

    U_P = layers.concatenate([V_P, self_fusion], axis=2)
    U_P = Bidirectional(LSTM(hidden_unit,return_sequences=True))(U_P)
    U_P = Dropout(.4)(U_P)

    start = gamma_s(U_P)
    start = layers.dot([start, UQ], axes=2)
    start = Reshape((max_para,))(start)
    start = layers.Activation("softmax")(start)

    end = gamma_e(U_P)
    end = layers.dot([end, UQ], axes=2)
    end = Reshape((max_para,))(end)
    end = layers.Activation("softmax")(end)

    model = Model([P, Q, P_exact_match],[start, end])
    if model_weights != None:
        model.load_weights(model_weights)

    # In[29]:

    model.compile(optimizer="adamax",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# In[ ]:
