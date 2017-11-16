from model import *

def train(_model, train_file, dictionary, model_weights, model_weights_save, batch_size, epoch) :
    ids, contexts, questions, answers_text, answers_start = get_train_data(train_file)
    d = get_dictionary(dictionary)
    # train_len = 10
    train_len = len(ids)
    paras, exact_match, qns = embeddings(d, train_len, ids, contexts, questions)
    ans_s, ans_e = answers(train_len, ids, contexts, answers_text, answers_start)
    if _model == "drqa":
        print('Creating DrQA model')
        model = DrQA(model_weights)
        model.fit([paras, qns, exact_match], [ans_s, ans_e],
                  batch_size=batch_size, epochs=epoch, validation_split=0.1)
    elif _model == "coa_hmn":
        print('Creating CoA HMN model')
        model = coa_hmn(model_weights)
        h = np.zeros((train_len,600,))
        c = np.zeros((train_len, 600))
        r = np.zeros((train_len,3,2*300))
        model.fit([paras, qns, h, c, r], [ans_s, ans_e],
                batch_size=batch_size, epochs=epoch,
                validation_split = 0.1)
    elif _model == "coa_res":
        print('Creating CoA Res model')
        model = coa_res(model_weights)
        model.fit([paras, qns], [ans_s, ans_e],
                batch_size=batch_size, epochs=epoch, validation_split=0.1)
    else:
        print('Creating Fusion model')
        model = Fusion(model_weights)
        model.fit([paras, qns, exact_match], [ans_s, ans_e],
                  batch_size=batch_size, epochs=epoch, validation_split=0.1)

    model.save(model_weights_save)
train("coa_res", "train.json", "glove/glove.6B.300d.txt","CoA_Res_report_2.h5","coa_res.h5",10,1)