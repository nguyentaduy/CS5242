from model import *

def train(_model, train_file, dictionary, model_weights, model_weights_save, batch_size, epoch) :
    ids, contexts, questions, answers_text, answers_start = get_train_data(train_file)
    d = get_dictionary(dictionary)
    # train_len = 500
    train_len = len(ids)
    paras, exact_match, qns = embeddings(d, train_len, ids, contexts, questions)
    ans_s, ans_e = answers(train_len, ids, contexts, answers_text, answers_start)
    if _model == "drqa":
        model = DrQA(model_weights)
        model.fit([paras, qns, exact_match], [ans_s, ans_e],
                  batch_size=batch_size, epochs=epoch, validation_split=0.1)
    else:
        model = Fusion(model_weights)
        model.fit([paras, qns, exact_match], [ans_s, ans_e],
                  batch_size=batch_size, epochs=epoch, validation_split=0.1)

        model.save(model_weights_save)
train("drqa", "../train.json", "../glove/glove.6B.300d.txt","../drqa.h5","../drqa.h5",256,2)
