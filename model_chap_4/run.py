import os, sys, argparse, torch, random, json, glob, pickle, re, tqdm 
from os.path import abspath, dirname, join, exists
from os import makedirs
from sys import stdout
import numpy as np
from copy import deepcopy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from collections import defaultdict

from model import Model
from event_dataset import Event_Dataset
from collate import collate
from constants import ind2label, label2ind
from evaluator import Evaluator
EVA = Evaluator()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
soft = torch.nn.Softmax(dim=1)

# def get_original_dataset():
def get_weights_for_training(last_pseudo_label, last_weights):
    weights = {}
    for e in last_pseudo_label.keys():
        if (last_pseudo_label[e] in ["gold","pos","neg","neu"]):
            weights[e] = last_weights[e]
        elif (last_pseudo_label[e][0] != last_pseudo_label[e][1]):
            weights[e] = last_weights[e] * 0.5
        else:
            weights[e] = last_weights[e] * 1.5
            if (weights[e] >= 1):
                weights[e] = 1
    return weights

def get_ds_events(levent2label):
    nes = {}
    with open("data/contexts_with_ds_ne_10_each_remove_ly/generated_contexts_label_proof.json", "r") as f:
        event2label = json.load(f)
    # with open("data/processed_labeled_data/all_data/labeled_events.json", "r") as f:
    #     all_labeled = json.load(f)
    for e in list(event2label.keys()):
        if e in levent2label.keys():
            for i in range(10):
                ne = event2label[e][i]
                # ne = remove_last_ly_word(ne)
                if (ne not in nes.keys() and ne not in levent2label.keys()):
                    nes[ne] = levent2label[e]
                # if (ne not in levent2label.keys() and ne in all_labeled.keys()):
                #     nes[ne] = all_labeled[e]
                # elif (nes[ne] != event2label_ori[e]):
                    # print("error: "+ ne)
                    # print(nes[ne])
                    # print(event2label_ori[e])
    with open("data/contexts_with_ds_ne_10_each_remove_ly/generated_events.json", "r") as f:
        event_sentiments = json.load(f)
    for e in nes.keys():
        # if (e in event_sentiments.keys() and e not in all_labeled.keys()):
        nes[e] = event_sentiments[e]
    nes_len = len(nes)
    nes_pos = {k: v for k, v in nes.items() if v == "pos"}
    nes_pos = {k: nes_pos[k] for k in sorted(nes_pos.keys())}
    nes_neg = {k: v for k, v in nes.items() if v == "neg"}
    nes_neg = {k: nes_neg[k] for k in sorted(nes_neg.keys())}
    nes_neu = {k: v for k, v in nes.items() if v == "neu"}
    nes_neu = {k: nes_neu[k] for k in sorted(nes_neu.keys())}
    nes_len = [int(len(nes_pos)/29),int(len(nes_neg)/23),int(len(nes_neu)/48)]
    smallest = min(nes_len)
    smallest = min(55, smallest)
    selected_keys = random.sample(list(nes_pos.keys()), smallest*29)
    nes_pos = {k: nes_pos[k] for k in selected_keys}
    selected_keys = random.sample(list(nes_neg.keys()), smallest*23)
    nes_neg = {k: nes_neg[k] for k in selected_keys}
    selected_keys = random.sample(list(nes_neu.keys()), smallest*48)
    nes_neu = {k: nes_neu[k] for k in selected_keys}
    nes = {}
    nes.update(nes_pos)
    nes.update(nes_neg)
    nes.update(nes_neu)
    nes = dict(sorted(nes.items()))
    # data = {
    #     "events": list(nes.keys()), 
    #     "labels": list(nes.values()), # hard labels, each item is a label_str
    # }
    return nes

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def remove_last_ly_word(sentence):
    words = sentence.split()
    if words and words[-1].endswith('ly'):
        words.pop()
    return ' '.join(words).strip()

def get_prototype(current_label_list, e2id, levent2label, model):
    # levents = list(sorted(levent2label.keys()))
    with open(join(args.train_dir, "labeled_events.json")) as f:
        event2label_ori = json.load(f)
    # with open("data/unlabeled_data/unlabeled_event2sentis.json") as f:
    #     original_train = json.load(f)
    # original_train.update(event2label_ori)
    original_train = event2label_ori
    originals = list(original_train.keys())
    levents = []
    for e in current_label_list:
        if (e in originals):
            levents.append(e)
    cevents, te2id = get_c2id(levents, args.contexts_num, args.contexts_dir)
    data = {
        "events": [e for e in levents for i in range(args.contexts_num)], 
        "labels": [levent2label[e] for e in levents for i in range(args.contexts_num)], # hard labels, each item is a label_str 
        "input_ids": [e2id[e] for e in levents for i in range(args.contexts_num)],
        "contexts_ids": te2id
        }
    prototypeloader = DataLoader(Event_Dataset(data), batch_size = args.batch_size, shuffle = args.to_shuffle, collate_fn = collate)
    
    labeled_representations = [[],[],[]]
    labeled_representations_centers = defaultdict(list)
    # labeled_representations_for_senti = {}
    # # unlabeled_representations_for_senti = {}
    with torch.no_grad():    
        for batch_item in tqdm.tqdm(prototypeloader, desc = "calculating prototype"):
            logits, event_embs = model.forward(batch_item)
            # event_embs = event_embs.detach().cpu()
            # probs = soft(logits)
            for i in range(len(list(batch_item["labels"]))):
                labeled_representations[list(batch_item["labels"])[i]].append(event_embs[i])
                # labeled_representations_for_senti[batch_item["events"][i]] = probs[i]
            # labeled_representations[batch_item["labels"].item()].append(event_embs)
        for i in range(len(labeled_representations)):
            labeled_representations[i] = torch.stack(labeled_representations[i]).detach().cpu()
            labeled_representations_centers[i] = labeled_representations[i].mean(dim=0)
    # print(labeled_representations_centers[0])
    # print(labeled_representations_centers[1])
    # print(labeled_representations_centers[2])
            
    # with open("/data1/chenpr/codes/nlp/A3+style_2/codes/data/processed_labeled_data/all_data/labeled_events.json", "r") as f:
    #     event2label1 = json.load(f)
    # with open("/data1/chenpr/codes/nlp/A3+style_2/codes/data/unlabeled_data/unlabeled_event2sentis.json", "r") as f:
    #     event2label2 = json.load(f)
    # event2label1.update(event2label2)

    # return (labeled_representations_centers, event2label1.keys())
    return (labeled_representations_centers, levents)
    
def get_c2id(events, num, contexts_dir):
    with open(contexts_dir + "/generated_contexts.json", "r") as f: # read only
        cevents = json.load(f)
        cevents = [cevents[e][0:int(num)] for e in events]
        cevents = [i for item in cevents for i in item] # flatten
    with open(contexts_dir + "/contexted_e2id.json", "r") as f: # read only
        te2id = json.load(f)
        te2id = [te2id[e][0:int(num)] for e in events]
        te2id = [i for item in te2id for i in item]
    return (cevents, te2id)

def get_c2e(contexts_dir):
    c2e_file =  contexts_dir + "/contexts2event.json"
    return c2e_file

def predict(model, test_dataloader):
    # use a trained model to make predictions over data
    model = model.to(device)
    model.eval()
    
    Y = [] # truelabel
    pred_Y = [] # predictlabel
    events = []
    pred_probs = []
    
    loss_funct = torch.nn.CrossEntropyLoss() ##
    avg_loss = 0 ##
    num_batches = 0 ##

    with torch.no_grad():      
        num_batches = len(test_dataloader)         
        for batch_item in tqdm.tqdm(test_dataloader, desc = "Making predcitions"):
            logits = model.forward(batch_item)[0]
            if(batch_item["labels"][0] != -1): ##
                loss = loss_funct(logits, batch_item["labels"].to(device)) ##
                avg_loss += float(loss.item()) ##
            probs = soft(logits).cpu().tolist()
            pred_probs.extend(probs)
            pred_Y.extend(np.argmax(probs, 1)) # 
            Y.extend(batch_item['labels'].tolist())
            events.extend(batch_item["events"])
    
        if(avg_loss): 
            avg_loss /= num_batches 
            print(f"average validation loss: {avg_loss}")

        # # major_voting ### 
        pred_Y = [pred_Y[i:i+args.contexts_num] for i in range(0,len(pred_Y),args.contexts_num)]
        tmp = []
        for s in pred_Y:
            count = [0, 0, 0]
            for l in s:
                if (l == 0): # pos
                    count[0] = count[0] + 1
                elif (l == 1): # neg
                    count[1] = count[1] + 1
                elif (l == 2 ): # neu
                    count[2] = count[2] + 1
            if (count[0] >= 4):
                tmp.append(0)
            elif (count[1] >= 4):
                tmp.append(1)
            elif (count[2] >= 4):
                tmp.append(2)
            else: # has to be [1/2/2]
                tmp.append(2) # neu
        pred_Ys = tmp

        pred_probs = [pred_probs[i:i+args.contexts_num] for i in range(0,len(pred_probs),args.contexts_num)]
        tmp = []
        for i, p in enumerate(pred_probs):
            tmp2 = []
            for j, ps in enumerate(p):
                if (pred_Y[i][j] == pred_Ys[i]):
                    tmp2.append(ps)
            if (len(tmp2) >= 4):
                tmp2 = np.array(tmp2)
                tmp.append(list(tmp2.mean(axis=0)))
            else:
                tmp.append([0,0,0])
        pred_probs = tmp
        
        events = [events[i] for i in range(0,len(events),args.contexts_num)]
        Y = [Y[i] for i in range(0,len(Y),args.contexts_num)]
        # # major_voting_end

    return events, pred_probs, pred_Ys, Y
    # return events, pred_probs, pred_Y, Y

def predict_wo_contexts(model, test_dataloader):
    # use a trained model to make predictions over data
    model = model.to(device)
    model.eval()
    
    Y = [] # 真实标签
    pred_Y = [] # 预测标签
    events = []
    pred_probs = []
    
    loss_funct = torch.nn.CrossEntropyLoss() ##
    avg_loss = 0 ##
    num_batches = 0 ##

    with torch.no_grad():      
        num_batches = len(test_dataloader)         
        for batch_item in tqdm.tqdm(test_dataloader, desc = "Making predcitions"):
            logits = model.forward(batch_item)[0]
            if(batch_item["labels"][0] != -1): ##
                loss = loss_funct(logits, batch_item["labels"].to(device)) ##
                avg_loss += float(loss.item()) ##
            probs = soft(logits).cpu().tolist()
            pred_probs.extend(probs)
            pred_Y.extend(np.argmax(probs, 1)) # 得到预测标签，取概率值最大的那个索引
            Y.extend(batch_item['labels'].tolist())
            events.extend(batch_item["events"])
    
        if(avg_loss): ##
            avg_loss /= num_batches ##
            print(f"average validation loss: {avg_loss}")

        # major_voting ### 这里还得改，参数要根据args.contexts_num来
        # pred_Y = [pred_Y[i:i+args.contexts_num] for i in range(0,len(pred_Y),args.contexts_num)]
        # tmp = []
        # for s in pred_Y:
        #     count = [0, 0, 0]
        #     for l in s:
        #         if (l == 0): # pos
        #             count[0] = count[0] + 1
        #         elif (l == 1): # neg
        #             count[1] = count[1] + 1
        #         elif (l == 2 ): # neu
        #             count[2] = count[2] + 1
        #     if (count[0] >= 4):
        #         tmp.append(0)
        #     elif (count[1] >= 4):
        #         tmp.append(1)
        #     elif (count[2] >= 4):
        #         tmp.append(2)
        #     else: # has to be [1/2/2]
        #         tmp.append(2) # neu
        # pred_Ys = tmp

        # pred_probs = [pred_probs[i:i+args.contexts_num] for i in range(0,len(pred_probs),args.contexts_num)]
        # tmp = []
        # for i, p in enumerate(pred_probs):
        #     tmp2 = []
        #     for j, ps in enumerate(p):
        #         if (pred_Y[i][j] == pred_Ys[i]):
        #             tmp2.append(ps)
        #     if (len(tmp2) >= 4):
        #         tmp2 = np.array(tmp2)
        #         tmp.append(list(tmp2.mean(axis=0)))
        #     else:
        #         tmp.append([0,0,0])
        # pred_probs = tmp
        
        # events = [events[i] for i in range(0,len(events),args.contexts_num)]
        # Y = [Y[i] for i in range(0,len(Y),args.contexts_num)]
        # major_voting_end

    # return events, pred_probs, pred_Ys, Y
    return events, pred_probs, pred_Y, Y


def evaluate(model, val_dataloader):
    # evaluate a model over labeled data 
    print(" >>> EVALUATION")
    events, pred_probs, y_pred, y_true = predict(model, val_dataloader)

    # ind2label = {0:"pos", 1:"neg", 2: "neu"}
    individual_eval, total_eval = EVA.eval(y_true, y_pred, ind2label)
    f1 = total_eval[-1]

    eval_str,_ = EVA.print_eval((individual_eval, total_eval), ind2label, None)
    # 计算混淆矩阵，行是真实标签，列是预测标签
    # __________|predict pos|predict neg|predict neu|
    # actual pos|    ...    |    ...    |    ...    |
    # actual neg|    ...    |    ...    |    ...    |
    # actual neu|    ...    |    ...    |    ...    |
    cnf_mat_str = EVA.text_confusion_matrix(y_true, y_pred, ind2label)

    return eval_str, cnf_mat_str, f1 

def evaluate_wo_contexts(model, val_dataloader):
    # evaluate a model over labeled data 
    print(" >>> EVALUATION")
    events, pred_probs, y_pred, y_true = predict_wo_contexts(model, val_dataloader)

    # ind2label = {0:"pos", 1:"neg", 2: "neu"}
    individual_eval, total_eval = EVA.eval(y_true, y_pred, ind2label)
    f1 = total_eval[-1]

    eval_str,_ = EVA.print_eval((individual_eval, total_eval), ind2label, None)
    # 计算混淆矩阵，行是真实标签，列是预测标签
    # __________|predict pos|predict neg|predict neu|
    # actual pos|    ...    |    ...    |    ...    |
    # actual neg|    ...    |    ...    |    ...    |
    # actual neu|    ...    |    ...    |    ...    |
    cnf_mat_str = EVA.text_confusion_matrix(y_true, y_pred, ind2label)

    return eval_str, cnf_mat_str, f1 

def load_train_vars(model_dir):
    train_vars_file = join(model_dir, 'train_config.json')
    with open(train_vars_file) as f:
        train_vars = json.load(f)
    return train_vars

def test_model(model_dir, test_data_infile):
    # test a model over labeled test data. This includes loading the model and loading the data
    train_vars = load_train_vars(model_dir)
    model_type = train_vars['model_type']
    local_model_type = "/data1/chenpr/huggingface/" + model_type # load from local
    tokenizer = AutoTokenizer.from_pretrained(local_model_type)
    
    model = Model(model_type)
    model_file = join(model_dir, 'model.ckpt')
    model.load_state_dict(torch.load(model_file)) # 加载模型参数

    print(f"Making test dataset ")
    with open(test_data_infile, "r") as f:
        event2label = json.load(f)
    events = list(event2label.keys())
    # labels = [event2label[e] for e in events for i in range(args.contexts_num)]
    # cevents, te2id = get_c2id(events, args.contexts_num, args.contexts_dir)
    # data = {
    #     "events":cevents, 
    #     "labels": [event2label[e] for e in events for i in range(args.contexts_num)],
    #     "input_ids": te2id
    # }
    # input_ids = [tokenizer.encode(e) for e in events for i in range(args.contexts_num)]
    # data = {
    #     "events": [e for e in events for i in range(args.contexts_num)], 
    #     "labels": labels, # hard labels, each item is a label_str 
    #     "input_ids": input_ids,
    #     "contexts_ids": te2id
    #     }
    labels = [event2label[e] for e in events]
    input_ids = [tokenizer.encode(e) for e in events]
    data = {
        "events": events, 
        "labels": labels, # hard labels, each item is a label_str 
        "input_ids": input_ids,
        }
    dataset = Event_Dataset(data, shuffle = False)
    test_dataloader = DataLoader(dataset, batch_size = args.test_batch_size, shuffle = False, collate_fn = collate)
    return evaluate_wo_contexts(model, test_dataloader)


def train(model, train_dataloader, val_dataloader, current_label_list, e2id, levent2label, args, ith_iter = None):
    # train a model over train data
    model_save_dir = args.model_save_dir + "/iter_" + str(ith_iter)
    if not exists(model_save_dir):
        makedirs(model_save_dir)
    model_file = join(model_save_dir, "model.ckpt")
    num_batches = len(train_dataloader) # 对于dataloader来说,len()返回的是dataloader中包含的batch数量
    
    model.to(device)
    """ optimizer """
    param_optimizer = list(model.named_parameters()) 
    # model.named_parameters()：返回一个列表，元素是元组，[(层参数名，层参数值),(..., ...), ...]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
               {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
               {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
          ]
    optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.lr, correct_bias=False)

    """ scheduler """
    num_training_steps = len(train_dataloader) * args.epoch
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler
    best_f1 = None
    loss_funct = torch.nn.CrossEntropyLoss(reduction='none')
    eval_str = ""
    for epoch in tqdm.tqdm(range(args.epoch), desc = f"Epochs {f'of {ith_iter}th iteration' if ith_iter is not None else ''}"): # i^th iteration: 0-9
        avg_loss = 0
        model.train()
        ## proto
        # labeled_representations_centers, original_set = get_prototype(current_label_list, e2id, levent2label, model)
        for batch_i, train_item in tqdm.tqdm(enumerate(train_dataloader), total = len(train_dataloader), desc = "Training batch"):
            if batch_i % 500 == 0:
                labeled_representations_centers, original_set = get_prototype(current_label_list, e2id, levent2label, model)
            logits = model.forward2(train_item, labeled_representations_centers, original_set)[0]
            # logits = model.forward(train_item)[0]
            loss = loss_funct(logits, train_item["labels"].to(device)) # batch size
            # print(loss)
            weights = torch.tensor(train_item["weights"]).to(device)
            loss = loss * weights
            loss = loss.sum() / weights.sum()
            avg_loss += float(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        avg_loss /= num_batches
        print(f"EPOCH {epoch} average training loss: {avg_loss}")

        if val_dataloader:
            print(f"----Evaluation----")
            cur_eval_str, cnf_mat_str, f1 = evaluate_wo_contexts(model, val_dataloader)
            print(f"Previous best F1 = {best_f1 if best_f1 is not None else 'None'}") # train的时候best_f1就是none

            if best_f1 is None or f1 > best_f1:
                best_f1 = f1
                eval_str = cur_eval_str
                torch.save(model.state_dict(), model_file) # 保存多个epoch中在验证集上最佳的模型
        else:
            torch.save(model.state_dict(), model_file)
    
    train_var_file = join(model_save_dir, 'train_config.json')
    with open(train_var_file, 'w') as f:
        json.dump(vars(args), f) # 保存模型参数，vars(args)将args对象的所有属性转换成字典的函数调用。

    print(f"Model saved to {model_file}. Training config saved to {train_var_file}")
    return model_file, eval_str

def do_training(args):
    # function to run DEST and train a model
    if not exists(args.model_save_dir):
        makedirs(args.model_save_dir)
    """ ================ make dataset ================ """
    model_type = "/data1/chenpr/huggingface/" + args.model_type # load from local
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    if not exists(f'./temp/{args.model_type}_e2id.json'):
        e2id = {} # map event to input ids
        # by bert tokenizer, add [CLS](id:101) at the start and [SEP](id:102) in the end
    else:
        print("Loading e2id from cache")
        e2id = json.load(open(f'./temp/{args.model_type}_e2id.json'))

    ''' 
        default = "data/unlabeled_data/senti2polar.json"
        contains the polarity scores for coreferent sentiment expressions
        ["true comedy", ["0.78858334", "0.0037742234", "0.2076424"]
        [sentiment expression, polarity score:[Positive, Negative,Neutral]]
        To generate polarity scores for these sentiment expressions
        we fine-tuned a BERT-based-uncased model over the training data in 
        SemEval 2017 Task 4: Sentiment Analysis in Twitter
    '''
    senti_polar = json.load(open(args.senti2polar)) # sentiment expression to polarity score 
    senti2polar = {}
    for senti, polar in senti_polar:
        '''
            senti_polar type: list
            senti_polar[0]:
            [senti='mash', polar=['0.50965345', '0.012091589', '0.47825494']]
        '''
        senti2polar[senti] = [float(i) for i in polar] # polar str2float

    print("Making training data")
    # args.train_dir=data/processed_labeled_data/10folds/fold"$fold"/train
    with open(join(args.train_dir, 'labeled_events.json')) as f:
        levent2label = json.load(f) # map labeled event 2 label in the training set 
    
    nes = get_ds_events(levent2label)
    events = list(nes.keys()) # keys: values => events: sentiment
    input_ids = [tokenizer.encode(e) for e in events for i in range(args.contexts_num)]
    cevents, te2id = get_c2id(events, args.contexts_num, args.contexts_dir)
    data = {
        "events": [e for e in events for i in range(args.contexts_num)], 
        "input_ids": input_ids,
        "contexts_ids": te2id
    }
    dataset = Event_Dataset(data, shuffle = False)
    dsloaders = DataLoader(dataset, batch_size = args.test_batch_size, shuffle = False, collate_fn = collate)
    
    # last_pseudo_label: 用于储存上一轮的伪标签
    last_pseudo_label = deepcopy(nes)
    no_pseudo_label = deepcopy(levent2label)
    for e in no_pseudo_label.keys():
        no_pseudo_label[e] = "gold"
    # for e in last_pseudo_label.keys(): # 对ds数据加额外的限制
    #     last_pseudo_label[e] = "ds_" + last_pseudo_label[e]
    last_pseudo_label.update(no_pseudo_label)
    
    levent2label.update(nes)
    # levent2label2 = nes
    # levent2label.update(levent2label2)
    #########
    assert len(levent2label)
    for e in levent2label: # 字典的in方法：查找键
        if e not in e2id:
            e2id[e] = tokenizer.encode(e) # by bert tokenizer
            # automatically add [CLS](id:101) at the start and [SEP](id:102) in the end
    print(f"{len(levent2label)} events in the training set")
    
    dataloaders = {
        "dev":None,
        'unlabel': None,
    }
    if args.dev_dir is not None:         
        print(f"Making dev dataset ")
        with open(join(args.dev_dir, "labeled_events.json"), "r") as f: # read only
            event2label = json.load(f)
        events = list(event2label.keys()) # keys: values => events: sentiment
        labels = [event2label[e] for e in events]
        # cevents, te2id = get_c2id(events, args.contexts_num, args.contexts_dir)
        # input_ids = [tokenizer.encode(e) for e in events for i in range(args.contexts_num)]
        # data = {
        #     "events":cevents,
        #     "labels": [event2label[e] for e in events for i in range(args.contexts_num)],
        #     "input_ids": input_ids,
        #     "contexts_ids": te2id,
        # }
        # for e in events:
        #     if e not in e2id:
        #         e2id[e] = tokenizer.encode(e)
        input_ids = [tokenizer.encode(e) for e in events]
        data = {
            "events": events, 
            "labels": labels , # hard labels, each item is a label_str 
            "input_ids": input_ids,
            }
        dataset = Event_Dataset(data, shuffle = False)
        dataloaders['dev'] = DataLoader(dataset, batch_size = args.test_batch_size, shuffle = False, collate_fn = collate) # 验证集，不必shuffle
        print(f"{len(events)} events in the development set")
    
    #unlabeled dataset
    uevent2polar = {}
    if args.unlabeled_event2sentis:
        '''
            default = "data/unlabeled_data/unlabeled_event2sentis.json"
            contains unlabeled events with their coreferent sentiment expressions collected from TWITTER
            {event: [sentiment_1, sentiment_2, ...], ...}
        '''
        uevent2sentis = json.load(open(args.unlabeled_event2sentis))
        for e in uevent2sentis:
            if e in levent2label: # levent2label: map labeled event 2 label in the training set 
                # for this event is alread labeled
                continue 
            polars = [] 
            for senti in set(uevent2sentis[e]): # list to set，避免处理重复的sentiment
                polars.append(senti2polar[senti]) # sentiment to score: [pos, neg, neu], score通过训练好的bert得到
            avg_polar = np.mean(polars, axis = 0) # class: np.ndarray
            uevent2polar[e] = avg_polar.tolist() # 为每一个事件得到：[pos, neg, neu]
        missing_uevents = [e for e in uevent2polar if e not in e2id]
        for e in tqdm.tqdm(missing_uevents, desc = "Encoding unlabeled events"):
            if e not in e2id: # 编码没编码过（没缓存过的）的unlabeled events
                e2id[e] = tokenizer.encode(e)
    
    if not exists('./temp'):
        os.makedirs('./temp')
    if not exists(f'./temp/{args.model_type}_e2id.json'):
        print("Saving e2id to cache")
        with open(f'./temp/{args.model_type}_e2id.json', 'w') as f:
            json.dump(e2id, f) # 将Python对象序列化为JSON格式的字符串并写入文件

    """ ================ train ================ """    
    print("Start training ") 
    eval_log = open(join(args.model_save_dir, 'eval_result_during_training.txt'), 'w')
    model_file = None
    sample_weights = deepcopy(levent2label)
    for e in sample_weights.keys():
        sample_weights[e] = 1
    for cyc in tqdm.tqdm(range(args.iter), desc = "DEST CYCLE"): # desc参数：为进度条设置描述文字
        sample_weights = get_weights_for_training(last_pseudo_label, sample_weights)
        for i in sample_weights.keys():
            if (sample_weights[i] < 1):
                print(i)
                print(sample_weights[i])
        levents = list(sorted(levent2label.keys())) # 给training_set里的event按照字母表顺序排序
        cevents, te2id = get_c2id(levents, args.contexts_num, args.contexts_dir)
        # data = {
        #     "events":cevents, 
        #     "labels": [levent2label[e] for e in levents for i in range(args.contexts_num)], # hard labels, each item is a label_str 
        #     "input_ids": te2id
        # }
        data = {
            "events": [e for e in levents for i in range(args.contexts_num)], 
            "labels": [levent2label[e] for e in levents for i in range(args.contexts_num)], # hard labels, each item is a label_str 
            "input_ids": [e2id[e] for e in levents for i in range(args.contexts_num)],
            "contexts_ids": te2id,
            "weights": [sample_weights[e] for e in levents for i in range(args.contexts_num)]
            }
        trainloader = DataLoader(Event_Dataset(data), batch_size = args.batch_size, shuffle = args.to_shuffle, collate_fn = collate)

        model = Model(args.model_type)
        # dev_eval_str：eval时指标的字符串表示形式
        model_file, dev_eval_str = train(model, trainloader, dataloaders["dev"], levents, e2id, levent2label, args, ith_iter = cyc )
        del model 
        torch.cuda.empty_cache()

        if dev_eval_str is not None:
            eval_log.write(f"CYC: {cyc}\nDEV\n{dev_eval_str}\n\n")
        
        count_c = {"pos":0,"neg":0,"neu":0}
        count = 0 
        if cyc != args.iter - 1: # cyc:0~9
            print("\nPredict over unlabeled data")
            model = Model(args.model_type)
            model.load_state_dict(torch.load(model_file)) # 加载使用labeled data训练好的模型

            # predict on unlabeled events
            uevent_list = list(uevent2polar.keys())
            input_ids = [e2id[e] for e in uevent_list for i in range(args.contexts_num)]
            cevents, te2id = get_c2id(uevent_list, args.contexts_num, args.contexts_dir)
            data = {
                "events": [e for e in uevent_list for i in range(args.contexts_num)], 
                "input_ids": input_ids,
                "contexts_ids": te2id
            }
            # input_ids = [e2id[e] for e in uevent_list]
            # data = {
            # "events":uevent_list, 
            # "input_ids": input_ids
            # }
            uloader = DataLoader(Event_Dataset(data, shuffle=False), batch_size = args.test_batch_size, shuffle = False, collate_fn = collate)
            
            pred_events, pred_probs, pred_Y, _ = predict(model, uloader) # predict() return events, pred_probs, pred_Y, Y
            for e, pred_prob, pred in zip(pred_events, pred_probs, pred_Y):
                label = ind2label[pred]

                # 情感极性得分高于一定值时，表示大概率是这个情感了，就把unlabeled转化成labeled
                if (e not in levent2label.keys()):
                    if pred_prob[pred] >= args.threshold:
                    # if pred_prob[pred] >= 0.1:
                        levent2label[e] = label
                        last_pseudo_label[e] = label
                        sample_weights[e] = 1
                        # uevent2polar.pop(e)
                        count += 1
                        count_c[label] += 1
                    elif pred_prob[2] >= args.neu_threshold:
                    # elif pred_prob[2] >= 0.1:
                        levent2label[e] = 'neu'
                        last_pseudo_label[e] = 'neu'
                        sample_weights[e] = 1
                        # uevent2polar.pop(e)
                        count += 1
                        count_c["neu"] += 1
                else: # e in levent2label.keys()
                    if pred_prob[pred] > 0.8: # 超有效标签
                        levent2label[e] = label
                    if pred_prob[pred] > 0.5: # 有效标签
                        if last_pseudo_label[e] in ["pos","neg","neu"]:
                            last_pseudo_label[e] = [last_pseudo_label[e], label]
                        else:
                            last_pseudo_label[e] = [last_pseudo_label[e][1], label]
                    # else:
                    #     sample_weights[e] =  sample_weights[e] * pred_prob[pred]

            # print(f"CYC {cyc}, learned {count} new events, {len(uevent2polar)} left in the unlabeled events")
            print(f"CYC {cyc}, learned {count} new events")
            print("specifically, " + str(count_c))

            pred_events, pred_probs, pred_Y, _ = predict(model, dsloaders) # predict() return events, pred_probs, pred_Y, Y
            for e, pred_prob, pred in zip(pred_events, pred_probs, pred_Y):
                label = ind2label[pred]

                # 情感极性得分高于一定值时，表示大概率是这个情感了，就把unlabeled转化成labeled
                if pred_prob[pred] > 0.8: # 超有效标签
                    levent2label[e] = label
                if pred_prob[pred] > 0.5: # 有效标签
                    if last_pseudo_label[e] in ["pos","neg","neu"]:
                        last_pseudo_label[e] = [last_pseudo_label[e], label]
                    else:
                        last_pseudo_label[e] = [last_pseudo_label[e][1], label]
                # else:
                #     sample_weights[e] =  sample_weights[e] * pred_prob[pred]

            del model 
            torch.cuda.empty_cache()
        if count == 0: # 学不到了（剩下的unlabeled都不太能确定极性）就结束训练
            print(f"Tranining stops at cyc {cyc}.")
            break 

    # if model_file is not None and args.test_dir:
    if args.test_dir:
        model_pos = model_file.split("iter")[0]
        for cyc in range(args.iter):
            model_file = model_pos + "iter_" + str(cyc) + "/model.ckpt"
            test_eval_str, test_cnf_mat_str, _ = test_model(dirname(model_file), join(args.test_dir, "labeled_events.json"))
            eval_log.write(f"CYC: {cyc}\nTEST\n{test_eval_str}\n{test_cnf_mat_str}\n\n")

    eval_log.close()

def get_args():
    """ ================= parse =============== """
    parser = argparse.ArgumentParser()
    # action指定了用户在命令行中使用--train参数时应该采取的动作
    # 'store_true'意味着包含--train参数时，该参数的值将被设置为True，否则将被设置为False
    parser.add_argument("--train", action = 'store_true', help = "If True, train a model")
    parser.add_argument("--test", action = 'store_true', help = 'If True, test a trained model over the testing data (labeled_events.json)')
    parser.add_argument("--predict", action = 'store_true', help = 'If True, predict over unseen data with a trained model')
    parser.add_argument("--checkpoint_dir", default = None, help = 'Directory that contains a model checkpoint to be used for the prediction or testing mode')

    # ------- argments for training or testing a DEST model. The following arguments would be used if --train is True ------
    parser.add_argument("--dev_dir", default = None, help = "dir that contains preprocessed labeled_events.json for development")
    parser.add_argument("--train_dir",default = None, help = "dir that contains preprocessed labeled_events.json for training")
    parser.add_argument("--test_dir", default = None, help = "dir that contains preprocessed labeled_events.json for testing")

    parser.add_argument("--unlabeled_event2sentis", default = "data/unlabeled_data/unlabeled_event2sentis.json", help = "the file containing a dictionary mapping an unlabeled event (key) to the corresponding set of coreferent sentiment expressions (value)")
    parser.add_argument("--senti2polar",  default = "data/unlabeled_data/senti2polar.json", help = "the file containing a dictionary that maps a sentiment expression (key) to the polarity score vector (values for positive, negative and neutral) produced by a sentiment classifier")
    parser.add_argument("--threshold", default = 0.95, type = float, help = "threshold used to select newly labeled event")
    parser.add_argument("--neu_threshold", default = 0.9, type = float, help = "threshold used to specificially select extra new neutral events")
    parser.add_argument("--iter", default = 10, type = int, help = "Maximum number of iterations of discourse-enhanced self-training")
    parser.add_argument("--test_batch_size", default = 100, type = int, help = "batch size during testing a model")
    parser.add_argument("--dropout", type = float, default = 0)
    parser.add_argument("--model_type", default = 'bert-base-uncased', choices = ["bert-base-uncased", "bert-base-cased", "bert-large-uncased", "bert-large-cased"])
    parser.add_argument("--seed", help = "seed", default = 100, type = int)
    parser.add_argument("--epoch", default = 5, type = int, help = "number epochs in training a model")
    parser.add_argument("--lr", default = 1e-5, type = float, help = "learning rate")
    parser.add_argument("--model_save_dir", default = "model_output", help = "Directory where the trained model would be saved")
    parser.add_argument("--batch_size", type = int, default = 50, help = "Training batch size")
    parser.add_argument("--to_shuffle", type = int, default = 1, choices = [1,0], help = "whether to shuffle the training data")
    parser.add_argument("--max_grad_norm", default = 1.00, type = float, help = "max gradient norm to clip")
    
    # ------- arguments for predicting over unseen data using a trained model ----
    parser.add_argument("--predict_str", default = None, help = " A string you want to make predictions for")
    parser.add_argument("--predict_infile", default = None, help = "A file containing lines of strings you want to make predictions for")
    parser.add_argument("--predict_outfile", default = None, help = "A file that saves the predicitons")

    parser.add_argument("--contexts_num", type = int, default = 5, help = "The number of contexts used for each sample")
    parser.add_argument("--contexts_dir", default = "data/contexts_with_senti", help = "Directory where the contexts are")
    # parser.add_argument("--gpu", default = None, help = "What kind of gpu and how much")

    # parser.add_argument("--prompt_tuning", action = 'store_true', help = "If True, use prompt tuning")
    # parser.add_argument("--prompt_tuning", action = 'store_true', help = "")

    args = parser.parse_args()
    
    # verify arguments, check if argument set
    if args.train: 
        assert args.unlabeled_event2sentis
        assert args.senti2polar
        assert args.train_dir
    
    if args.test:
        assert args.test_dir 
        assert args.checkpoint_dir

    if args.predict:
        assert args.checkpoint_dir
        assert args.predict_str is not None or args.predict_infile
    
    # constants: label2ind = {"pos": 0, "neg": 1, "neu": 2}
    args.num_classes = len(label2ind)

    return args
    
# def do_predict(checkpoint_dir, predict_str, predict_infile):
#     train_vars = load_train_vars(checkpoint_dir)
#     model_type = train_vars["model_type"]
#     model = Model(model_type)
#     model.load_state_dict(torch.load(join(checkpoint_dir, "model.ckpt")))

#     if predict_str is not None: 
#         strs = [predict_str] 
#     else:
#         with open(predict_infile) as f:
#             # 读取文件中的每一行（除去空行），去除行首尾的空白字符，然后将结果列表赋值给strs变量
#             strs = [line.strip() for line in f if line.strip() != ""]

#     model_type = "/data1/chenpr/huggingface/" + model_type # load from local
#     cevents, te2id = get_c2id(strs, 1, args.contexts_dir)
#     data = {
#         "events":cevents,
#         "input_ids": te2id
#     }
#     # tokenizer = AutoTokenizer.from_pretrained(model_type)
#     # input_ids = [tokenizer.encode(s) for s in strs]
#     # data = {
#     # "events":strs,
#     # "input_ids": input_ids
#     # }
#     uloader = DataLoader(Event_Dataset(data, shuffle = False),batch_size = args.test_batch_size, shuffle = False, collate_fn = collate)
    
#     pred_events, pred_probs, pred_Y, _ = predict(model, uloader)
#     ## pred_events = strs ##
#     # make scores into the 0-100 scale
#     pred_probs = [[k*100 for k in prob] for prob in pred_probs]
#     if predict_str:
#         # print out the one result
#         print(f"""
#             INPUT: {pred_events[0]}
#             OUTPUT: {ind2label[pred_Y[0]]}
#             SCORE: pos {pred_probs[0][label2ind['pos']]:.2f}, neg {pred_probs[0][label2ind['neg']]:.2f}, neu {pred_probs[0][label2ind['neu']]:.2f} 
#             """)

#     out_dict = [
#         (
#             e, 
#             {   'pos': pred_prob[label2ind['pos']], 
#                 'neg': pred_prob[label2ind['neg']],
#                 'neu': pred_prob[label2ind['neu']]
#             }, 
#             ind2label[pred_y]
#         ) for e, pred_prob, pred_y in zip(pred_events, pred_probs, pred_Y)
#         ]


#     if args.predict_infile:
#         predict_outfile = args.predict_infile + ".json" if args.predict_outfile is None else args.predict_outfile
#     else:
#         predict_outfile = "out.json"

#     with open(predict_outfile, 'w') as f:
#         json.dump(out_dict, f, indent = 2)

#     print(f"Predictions are saved to {predict_outfile}")

if __name__ == "__main__":

    args = get_args()
    print(f"{args}\n")

    # # 单卡训练
    # gpu = args.gpu
    # set_device(gpu)

    seed = args.seed
    seed_everything(seed)

    if args.train: 
        do_training(args)
    # if args.test: # 在do_training时其实就做过一次了，写这里完全是多余的
    #     print("Testing")
    #     test_model(args.checkpoint_dir, join(args.test_dir, "labeled_events.json"))
    # if args.predict: # 不重新训练，直接测试模型效果时才用到
    #     print("Predicting")
    #     do_predict(args.checkpoint_dir, args.predict_str, args.predict_infile)