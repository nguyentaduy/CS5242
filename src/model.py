
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
from keras.layers import Lambda, Activation,recurrent, Bidirectional, Dense, Flatten, Conv1D, Dropout
from keras.layers import LSTM, GRU, concatenate, multiply, add, Reshape, MaxPooling1D, BatchNormalization
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

#Functions for CoA-HMN model
def Maxout(x, num_unit=16):
    input_shape = x.get_shape().as_list()
   
    ch = input_shape[-1]

    x = K.reshape(x, (-1, input_shape[1], ch // num_unit, num_unit))
    x = K.max(x, axis=3)
    
    return x

def LSTM_D(hidden_unit, input_layer, h_state, c_state, Wi, Wf, Wc, Wo):

    i = concatenate([input_layer, h_state])
    i = Wi(i)
    i = Activation('sigmoid')(i)

    ft = Wf(i)
    ft = Activation('sigmoid')(i)

    Ct_ = Wc(i)
    Ct_ = Activation('tanh')(i)

    Ct1 = multiply([ft, c_state])
    Ct2 = multiply([i, Ct_])
    Ct = add([Ct1, Ct2])

    ot = Wo(i)

    ht = multiply([ot, Ct])

    return ht, Ct

def HMN(input_layer, r_layer, D1, D2, D3, DD):
    I = layers.concatenate([input_layer, r_layer], axis = 1)
    mt1 = D1(I)
    mt1 = Lambda(Maxout)(mt1)
    mt1 = Dropout(0.5)(mt1)
    mt2 = D2(mt1)
    mt2 = Lambda(Maxout)(mt2)
    mt2 = Dropout(0.5)(mt2)
    mt = layers.concatenate([mt1,mt2])
    mt = D3(mt)
    out = Lambda(Maxout)(mt)
    out = Flatten()(out)
    out = DD(out)
    out = Activation('softmax')(out)
    return out

def coa_hmn(model_weights = None):
    EMBED_HIDDEN_SIZE = 300
    hidden_unit = 600
    #shared weight layers
    shared_LSTM = LSTM(EMBED_HIDDEN_SIZE, return_sequences=True)
    shared_start = Dense(max_para, activation='softmax')
    shared_end = Dense(max_para, activation='softmax')
    Wi = Dense(hidden_unit)
    Wf = Dense(hidden_unit)
    Wc = Dense(hidden_unit)
    Wo = Dense(hidden_unit)
    D1a = Dense(640)
    D2a = Dense(640)
    D3a = Dense(16)
    DDa = Dense(max_para)
    D1b = Dense(640)
    D2b = Dense(640)
    D3b = Dense(16)
    DDb = Dense(max_para)
    Dr = Dense(2*EMBED_HIDDEN_SIZE)
    ###########################
    sentence = layers.Input(shape=(max_para,dimension), dtype='float32')
    encoded_sentence =shared_LSTM(sentence)

    question = layers.Input(shape=(max_q,dimension), dtype='float32')
    encoded_question = shared_LSTM(question)

    #Encoder
    merge_1 = layers.dot([encoded_sentence, encoded_question], axes = 2 )
    A_Q = layers.Activation("softmax")(merge_1)
    merge_2 = layers.dot([encoded_question, encoded_sentence], axes = 2 )
    A_D = layers.Activation("softmax")(merge_2)
    C_Q = layers.dot([A_Q, encoded_sentence], axes = 1 )

    C_Q = layers.concatenate([encoded_question, C_Q], axis=2)
    C_D = layers.dot([A_D, C_Q], axes=1)
    C_ = layers.concatenate([encoded_sentence, C_D], axis=2)

    U = Bidirectional(LSTM(EMBED_HIDDEN_SIZE, return_sequences=True))(C_)
    U = Dropout(0.5)(U)

    #Decoder
    h_state_i = layers.Input(shape=(hidden_unit,))
    h_state = Dense(hidden_unit, kernel_initializer = keras.initializers.Zeros())(h_state_i)

    C_state_i = layers.Input(shape=(hidden_unit,))
    C_state = Dense(hidden_unit, kernel_initializer = keras.initializers.Zeros())(C_state_i)

    r_layer_init = layers.Input(shape=(3,2*EMBED_HIDDEN_SIZE))
    r_layer = Dense(2*EMBED_HIDDEN_SIZE, kernel_initializer = keras.initializers.Zeros())(r_layer_init)

    start = HMN(U, r_layer, D1a, D2a, D3a, DDa)
    end = HMN(U, r_layer, D1b, D2b, D3b, DDb)

    #########################
    for i in range(5):
        start_comb = Reshape((max_para,1))(start)
        end_comb = Reshape((max_para,1))(end)
        combine_output = concatenate([start_comb,end_comb], axis = 2)
        LSTM_input_U = layers.dot([combine_output, U], axes = 1)
        LSTM_input = Reshape((2*600,))(LSTM_input_U)

        h_state, C_state = LSTM_D(hidden_unit, LSTM_input, h_state, C_state, Wi, Wf, Wc, Wo)
        h_reshape = Reshape((1,600))(h_state)
        r_layer = concatenate([LSTM_input_U, h_reshape], axis = 1)
        r_layer = Dr(r_layer)
        r_layer = Activation('tanh')(r_layer)

        start = HMN(U, r_layer, D1a, D2a, D3a, DDa)
        end = HMN(U, r_layer, D1b, D2b, D3b, DDb)
        
    model = Model([sentence, question, h_state_i, C_state_i, r_layer_init],[start, end])
    if model_weights != None:
        model.load_weights(model_weights)
    optimizer = optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#Functions for CoA-Res
#Convolution, batch norm, relu unit
def convBN_pool(input_layer, conv_channels):
    convlayer = Conv1D(conv_channels, 1, padding = 'valid', strides = 2)(input_layer)
    BN = BatchNormalization(axis=-1, momentum = 0.99, epsilon=0.001, center=True, scale = True, 
                            beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros')(convlayer)
    activation = layers.PReLU()(BN)

    return activation

def convBN(input_layer, conv_channels):
    convlayer = Conv1D(conv_channels, 3, padding = 'same')(input_layer)
    BN = BatchNormalization(axis=-1, momentum = 0.99, epsilon=0.001, center=True, scale = True, 
                            beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', 
                            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(convlayer)
    activation = layers.PReLU()(BN)

    return activation

def RU(input_layer, conv_channels, d_rate):
    # input tensor for a 3-channel 256x256 image
    x = input_layer
    # 3x3 conv with 3 output channels (same as input channels)
    y = Conv1D(conv_channels, 3, padding='same', dilation_rate = d_rate)(x)
    y = Conv1D(conv_channels, 3, padding='same', dilation_rate = d_rate)(y)
    # this returns x + y.
    z = layers.add([x, y])
    return z

def coa_res(model_weights = None):
    EMBED_HIDDEN_SIZE = 300
    #shared weight layers
    shared_LSTM = LSTM(EMBED_HIDDEN_SIZE, return_sequences=True)

    ###########################
    sentence = layers.Input(shape=(max_para,dimension), dtype='float32')
    encoded_sentence =shared_LSTM(sentence)

    question = layers.Input(shape=(max_q,dimension), dtype='float32')
    encoded_question = shared_LSTM(question)

    #Encoder
    merge_1 = layers.dot([encoded_sentence, encoded_question], axes = 2 )
    A_Q = layers.Activation("softmax")(merge_1)
    merge_2 = layers.dot([encoded_question, encoded_sentence], axes = 2 )
    A_D = layers.Activation("softmax")(merge_2)
    C_Q = layers.dot([A_Q, encoded_sentence], axes = 1 )

    C_Q = layers.concatenate([encoded_question, C_Q], axis=2)
    C_D = layers.dot([A_D, C_Q], axes=1)
    C_ = layers.concatenate([encoded_sentence, C_D], axis=2)

    U = Bidirectional(LSTM(EMBED_HIDDEN_SIZE, return_sequences=True))(C_)
    U = Dropout(0.5)(U)

    #Decoder
    start = convBN(U, 100)
    start = RU(start, 100, 1)
    start = Dropout(0.5)(start)
    start = convBN_pool(start, 64)
    start = RU(start, 64, 1)
    start = convBN_pool(start, 64)
    start = RU(start, 64, 2)
    start = convBN_pool(start, 128)
    start = RU(start, 128, 1)
    start =layers.PReLU()(start)
    start = RU(start, 128, 2)
    start = Dropout(0.5)(start)
    start = convBN_pool(start, 256)
    start = RU(start, 256, 2)
    start = Dropout(0.5)(start)
    start = convBN_pool(start, 128)
    start = RU(start, 128, 1)
    start = layers.PReLU()(start)
    start = Dropout(0.5)(start)
    start = convBN(start, 64)
    start = RU(start, 64, 1)
    start = layers.PReLU()(start)
    start = Flatten()(start)
    start = Dropout(0.5)(start)
    start = Dense(max_para, activation='softmax', name='output_1')(start)


    end = GRU(100, return_sequences=True)(U)
    end = convBN(end, 100)
    end = RU(end, 100, 1)
    end = Dropout(0.5)(end)
    end = convBN_pool(end, 64)
    end = RU(end, 64, 1)
    end = convBN_pool(end, 64)
    end = RU(end, 64, 2)
    end = convBN_pool(end, 128)
    end = RU(end, 128, 1)
    end = layers.PReLU()(end)
    end = RU(end, 128, 2)
    end = Dropout(0.5)(end)
    end = convBN_pool(end, 256)
    end = RU(end, 256, 2)
    end = Dropout(0.5)(end)
    end = convBN_pool(end, 128)
    end = RU(end, 128, 1)
    end = layers.PReLU()(end)
    end = Dropout(0.5)(end)
    end = convBN(end, 64)
    end = RU(end, 64, 1)
    end = layers.PReLU()(end)
    end = Flatten()(end)
    end = Dropout(0.5)(end)
    end = Dense(max_para, activation='softmax', name='output_2')(end)

    model = Model([sentence, question],[start, end])
    if model_weights != None:
        model.load_weights(model_weights)
    model.compile(optimizer='adam',
                  loss={'output_1': 'categorical_crossentropy', 'output_2': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    return model
