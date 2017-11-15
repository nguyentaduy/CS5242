
# coding: utf-8

# In[1]:

from __future__ import print_function
from nltk.tokenize import word_tokenize
import csv
from preprocess import *
from model import *

def predict(_model, model_weights, test_file, dictionary, out_file): #em Use exact match or not
    ids, contexts, questions = get_test_data(test_file)
    d = get_dictionary(dictionary)
    # test_len = 500
    test_len = len(ids)
    paras, exact_match, qns = embeddings(d, test_len, ids, contexts, questions)
    if _model == "drqa":
        model = DrQA(model_weights)
        out = model.predict([paras,qns,exact_match])
    else:
        model = Fusion(model_weights)
        out = model.predict([paras,qns,exact_match])
    out_1 = np.argmax(out[0], axis=1)
    out_2 = np.argmax(out[1], axis=1)

    for i in range(test_len):
        s = 0
        t = 1
        for j in range(max_para):
            for k in range(j, min(j+15, max_para)):
                if out[0][i][s] * out[1][i][t] < out[0][i][j] * out[1][i][k]:
                    s = j
                    t = k
        out_1[i] = s
        out_2[i] = t

    out_2 = out_2 + 1

    with open(out_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Id','Answer'])
        for i in range(test_len):
            a = search(contexts[ids[i]], out_1[i])
            b = search(contexts[ids[i]], out_2[i])
            writer.writerow([ids[i],normalize_answer(contexts[ids[i]][a:b])])
predict("drqa", "../drqa.h5", "../test.json", "../glove/glove.6B.300d.txt", "submission.csv")

# In[ ]:
