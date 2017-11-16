from __future__ import print_function
import json
from pprint import pprint
import numpy as np

import keras
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from keras import optimizers
from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import Lambda, Activation,recurrent, Bidirectional, Dense, Flatten, Conv1D, Dropout, LSTM, GRU, concatenate, multiply, add, Reshape, MaxPooling1D, BatchNormalization
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.engine.topology import Layer
import string
import re

# In[2]:


max_para = 600
max_q = 50
dimension = 300

def get_dictionary(vocab):
    d = dict()
    #Add encoding ='utf=8' if get DecodeError
    with open(vocab) as f:
        lines = f.readlines()
        for l in lines:
            values = l.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            d[word] = coefs
    return d

def create_one_hot_answer(para, answer, answer_start, option, max_length):
    if option == "s":
        from_begin = para[0:answer_start]
    else:
        from_begin = para[0:answer_start+len(answer)]
    l = len(word_tokenize(from_begin))
    one_hot = np.zeros(max_length)
    if option == "s":
        one_hot[min(max_para-1,l)] = 1
    else:
        one_hot[min(max_para-1,l-1)] = 1
    return one_hot

def get_train_data(data_filename):
    with open(data_filename) as data_file:
        data = json.load(data_file)
    ids = []
    contexts = dict()
    questions = dict()
    answers_text = dict()
    answers_start = dict()
    for i in range(len(data)):
        paragraphs = data[i]["paragraphs"]
        for j in range(len(paragraphs)):
            context = paragraphs[j]["context"]
            qas = paragraphs[j]["qas"]
            for k in range(len(qas)):
                id_ = qas[k]["id"]
                answer = qas[k]["answer"]
                question = qas[k]["question"]
                ids.append(id_)
                contexts[id_] = context
                answers_start[id_] = answer["answer_start"]
                answers_text[id_] = answer["text"]
                questions[id_] = question
    return ids, contexts, questions, answers_text, answers_start

def get_test_data(data_filename):
    with open(data_filename) as data_file:
        data = json.load(data_file)
    ids = []
    contexts = dict()
    questions = dict()
    for i in range(len(data)):
        paragraphs = data[i]["paragraphs"]
        for j in range(len(paragraphs)):
            context = paragraphs[j]["context"]
            qas = paragraphs[j]["qas"]
            for k in range(len(qas)):
                id_ = qas[k]["id"]
                question = qas[k]["question"]
                ids.append(id_)
                contexts[id_] = context
                questions[id_] = question
    return ids, contexts, questions


def embeddings(d, train_len, ids, contexts, questions, max_para=max_para, max_q=max_q, dimension=dimension):
    paras = np.zeros((train_len, max_para, dimension))
    qns = np.zeros((train_len, max_q, dimension))
    exact_match = np.zeros((train_len, max_para, 3))
    lmtzr = WordNetLemmatizer()

    for i in range(train_len):
        words = word_tokenize(contexts[ids[i]])
        qs = word_tokenize(questions[ids[i]])
        qs_1 = [w.lower() for w in qs]
        qs_2 = [lmtzr.lemmatize(w) for w in qs_1]
        for j in range(min(max_para - 1, len(words))):
            if words[j].lower() in d:
                paras[i][j] = d[words[j].lower()]
            if words[j] in qs:
                exact_match[i][j][0] = 1
            if words[j].lower() in qs_1:
                exact_match[i][j][1] = 1
            if lmtzr.lemmatize(words[j].lower()) in qs_2:
                exact_match[i][j][2] = 1
        for j in range(min(max_q - 1, len(qs))):
            if qs[j].lower() in d:
                qns[i][j] = d[qs[j].lower()]

    return paras, exact_match, qns
def answers(train_len, ids, contexts, answers_text, answers_start, max_para=max_para):

    ans_starts = [create_one_hot_answer(contexts[ids[i]], answers_text[ids[i]], answers_start[ids[i]], "s", max_para) for i in range(train_len)]
    ans_ends = [create_one_hot_answer(contexts[ids[i]], answers_text[ids[i]], answers_start[ids[i]], "e", max_para) for i in range(train_len)]

    ans_s = np.array(ans_starts)
    ans_e = np.array(ans_ends)
    return ans_s, ans_e

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def search(para, pos):
    word_list = word_tokenize(para)
    s = ''.join(word_list[:pos])
    if pos == 0:
        return 0
    if len(word_list) <= pos:
        return len(para)
    else:
        begin, end = 0, len(para)
        c = 0
        while begin < end and c < 1000:
            c += 1
            mid = (begin + end) // 2
            a = word_tokenize(para[:mid])
            s_ = ''.join(a)
            if (len(s_) < len(s)):
                begin = mid + 1
            elif (len(s_) > len(s)):
                end = mid
            elif s == s_:
                break
        return mid