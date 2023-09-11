import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import sample 
import random
from torch.nn.utils import clip_grad_norm_
import time
import pickle
import argparse
import numpy as np
from os import path
from data import *
from utils import *
from model import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print_msg(str(device))

rule_conf = {}
candidate_rule = {}
# 这里传入的max_path_len对采样速度影响极大
def sample_training_data(max_path_len, anchor_num, fact_rdf, entity2desced, head_rdict,
                         relation_dom_type, relation_ran_type, rel2idx, idx2rel, relation_num):
    print("Sampling training data...")
    anchors_rdf = []
    # per_anchor_num = anchor_num//((head_rdict.__len__() -1) //2)  # 10000除以12也就是relation的个数（12）等于833

    # 根据type权重采样anchor
    # weights = {}
    # for key, value in relation_dom_type.items():
    #     weight = len(value)
    #     weights[key] = weight

    print("Number of head relation:{}".format((head_rdict.__len__() -1) // 2))
    # print ("Number of per_anchor_num: {}".format(per_anchor_num))
    fact_dict = construct_fact_dict(fact_rdf)  # key是relation，value是此relation所有的fact
    TBox_ABox_proportion = 0.5
    weights_TBox = {}
    weights_TBox_sum = 0
    for key, value in relation_dom_type.items():
        weight_TBox = len(value)
        weights_TBox_sum += weight_TBox
        weights_TBox[key] = weight_TBox
    for key, value in relation_ran_type.items():
        weight_TBox = len(value)
        weights_TBox_sum += weight_TBox
        weights_TBox[key] += weight_TBox
    weights_TBox = {k: v / weights_TBox_sum for k, v in weights_TBox.items()}

    weights_ABox_sum = 0  # train facts tests valid都在
    weights_ABox = {}
    for key, value in fact_dict.items():
        weight_ABox = len(value)
        weights_ABox_sum += weight_ABox
        weights_ABox[rel2idx[key]] = weight_ABox
    weights_ABox = {k: v / weights_ABox_sum for k, v in weights_ABox.items()}


    for head in head_rdict.rel2idx:
        if head != "None" and "inv_" not in head:
            weight_head = weights_TBox[rel2idx[head]] * TBox_ABox_proportion + weights_ABox[rel2idx[head]] * (1 - TBox_ABox_proportion)
            per_anchor_num = round(anchor_num * weight_head)
            sampled_rdf = sample_anchor_rdf(fact_dict[head], num=per_anchor_num)  # 从此head的所有fact中随机抽取833个fact  # 可不可以根据频率来抽取，频率高的抽取多一点，频率低的抽取少一点
            anchors_rdf.extend(sampled_rdf)
    print ("Total_anchor_num",len(anchors_rdf))
    train_rule, train_rule_dict = [],{}
    len2train_rule_idx = {}  # 4:[tensor([ 0,  2,  8, 10, -1, 10]), tensor([ 0,  2,  8,  0, -1,  0])]这种，一共24个relation，每个取833个facts，一个facts挖出来十几个封闭路径，所以很多，debug要转很多时间，还显示不了
    sample_number = 0
    for anchor_rdf in anchors_rdf:
        rule_seq, record = construct_rule_seq(fact_rdf, anchor_rdf, entity2desced, max_path_len,
                                              relation_dom_type, relation_ran_type, rel2idx, idx2rel, relation_num, PRINT=False)
        sample_number += len(record)
        if len(rule_seq) > 0:
            train_rule += rule_seq
            for rule in rule_seq:
                idx = torch.LongTensor(rule2idx(rule, head_rdict))
                h = head_rdict.idx2rel[idx[-1].item()] # 这里的-1是最后一个元素不是间隔符
                if h not in train_rule_dict:
                    train_rule_dict[h] = []
                train_rule_dict[h].append(idx)
                # cluster rules according to its length
                body_len = len(idx) - 2
                if body_len in len2train_rule_idx.keys():
                    len2train_rule_idx[body_len] += [idx]
                else:
                    len2train_rule_idx[body_len] = [idx]
                    
    print ("# train_rule:{}".format(sample_number)) # 24个relation，每个relation取833个facts，每个facts挖出来十几个封闭路径
    print ("# head:{}".format(len(train_rule_dict)))
    for h in train_rule_dict:
        print ("head {}:{}".format(h,len(train_rule_dict[h])))
        
    rule_len_range = list(len2train_rule_idx.keys())
    print("Fact set number:{} Sample number:{}".format(len(fact_rdf), sample_number))
    for rule_len in rule_len_range:
        print("sampled examples for rule of length {}: {}".format(rule_len, len(len2train_rule_idx[rule_len])))
    return len2train_rule_idx


def train(args, dataset):
    rdict = dataset.get_relation_dict()
    head_rdict = dataset.get_head_relation_dict()
    # 这里的fact是facts.txt.inv,因为下面的main里面inv置为true了
    all_rdf = dataset.fact_rdf + dataset.train_rdf + dataset.valid_rdf
    entity2desced = construct_descendant(all_rdf)  # key是head, value是head对应的所有relation和tail
    relation_num = rdict.__len__()
    # Sample training data
    max_path_len = args.max_path_len
    anchor_num = args.anchor

    relation_dom_type = dataset.relation_dom_type
    relation_ran_type = dataset.relation_ran_type

    rel2idx = dataset.rdict.rel2idx
    idx2rel = dataset.rdict.idx2rel

    # 在图上进行深度优先搜索，获得训练用的path，后面输入model
    len2train_rule_idx = sample_training_data(max_path_len, anchor_num, all_rdf, entity2desced, head_rdict,
                                              relation_dom_type, relation_ran_type, rel2idx, idx2rel, relation_num)

    # sampling_path = sample_training_data(max_path_len, anchor_num, all_rdf, entity2desced, head_rdict)
    # # 因为上面的sample_training_data函数太慢了，我把它的结果保存下来，下次直接读取
    # sampling_path_save_path = "sampling_path"
    # if not path.isfile(sampling_path_save_path):
    #     with open(sampling_path_save_path, 'wb') as f:
    #         pickle.dump(sampling_path, f)
    # else:
    #     with open(sampling_path_save_path, 'rb') as f:
    #         len2train_rule_idx = pickle.load(f)

    print_msg("  Start training  ")
    # model parameter
    batch_size = args.batch_size  # 1000
    emb_size = 1024
    
    # train parameter
    n_epoch = args.n_epoch  # 1500
    lr = 0.000025
    
    body_len_range = list(range(2,max_path_len+1))
    print ("body_len_range",body_len_range)
    
    # model
    model = Encoder(relation_num, emb_size, device)
        
    if torch.cuda.is_available():
        model = model.cuda()
        
    # loss
    loss_func_head = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    """
    Training
    """
    model.train()
    start = time.time()
    train_acc = {}

    for rule_len in body_len_range:
        rule_ = len2train_rule_idx[rule_len]
        print("\nrule length:{}".format(rule_len))
        
        train_acc[rule_len] = []
        for epoch in range(n_epoch):
            model.zero_grad()
            if len(rule_) > batch_size:
                sample_rule_ = sample(rule_, batch_size) #[[17,21,-1,23],[2,23,-1,8],...]
            else:
                sample_rule_ = rule_
            body_ = [r_[0:-2] for r_ in sample_rule_] #[[17,21],[2,23],...]
            head_ = [r_[-1] for r_ in sample_rule_] #[23,8,...]

            inputs_h = body_
            targets_h = head_
            
            # stack list into Tensor
            inputs_h = torch.stack(inputs_h, 0).to(device) # 化为了一整个tensor
            targets_h = torch.stack(targets_h, 0).to(device)
            
            # forward pass 
            pred_head, _entropy_loss = model(inputs_h) # 输入模型进行预测，得到预测结果和熵损失
        
            # 这里损失函数改一下
            loss_head = loss_func_head(pred_head, targets_h.reshape(-1))  # 计算当前batch的head预测结果和真实结果之间的交叉熵损失
            
            entropy_loss = _entropy_loss.mean()
        
            loss = args.alpha * loss_head + (1-args.alpha) * entropy_loss

            
            if epoch % (n_epoch//10) == 0:
                print("### epoch:{}\tloss_head:{:.3}\tentropy_loss:{:.3}\tloss:{:.3}\t".format(epoch, loss_head, entropy_loss,loss))
                
            train_acc[rule_len].append(((pred_head.argmax(dim=1) == targets_h.reshape(-1)).sum() / pred_head.shape[0]).cpu().numpy())
            
            # backward and optimize
            clip_grad_norm_(model.parameters(), 0.5)
            loss.backward()
            optimizer.step()

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("LogicFormer Epoch vs Accurary")
        train_acc[rule_len] = [float(x) for x in train_acc[rule_len]]
        plt.plot(train_acc[rule_len])
        plt.savefig('../figures/{}_{}.png'.format(args.model,rule_len))
        
    end = time.time()
    print("Time usage: {:.2}".format(end - start))
        
    print("Saving model...")
    with open('../results/{}'.format(args.model), 'wb') as g:
        pickle.dump(model, g)
        

def enumerate_body(relation_num, rdict, body_len):
    import itertools
    all_body_idx = list(list(x) for x in itertools.product(range(relation_num), repeat=body_len))
    # transfer index to relation name
    idx2rel = rdict.idx2rel
    all_body = []
    for b_idx_ in all_body_idx:
        b_ = [idx2rel[x] for x in b_idx_]
        all_body.append(b_)
    return all_body_idx, all_body


def test(args, dataset):
    head_rdict = dataset.get_head_relation_dict()
    with open('../results/{}'.format(args.model), 'rb') as g:
        if torch.cuda.is_available():
            model = pickle.load(g)
            model.to(device)
        else:
            model = torch.load(g, map_location='cpu')
    print_msg("  Start Eval  ")
    model.eval()    
    #body_list = ['brother|bro|brother|daughter'] 
    r_num = head_rdict.__len__()-1
    
    # model parameter
    batch_size = 1000
    
    rule_len = args.learned_path_len
    print("\nmax rule length:{}".format(rule_len))

    for rule_len in range(2, args.learned_path_len+1):
        probs = []
        _, body = enumerate_body(r_num, head_rdict, body_len=rule_len)  # 把所有可能的body都列出来
        body_list = ["|".join(b) for b in body]  # 给中间加上竖线
        candidate_rule[rule_len] = body_list
        n_epoches = math.ceil(float(len(body_list))/ batch_size)
        for epoches in range(n_epoches):
            bodies = body_list[epoches: (epoches+1)*batch_size]
            if epoches == n_epoches-1:
                bodies = body_list[epoches*batch_size:]
            else:
                bodies = body_list[epoches*batch_size: (epoches+1)*batch_size]

            body_idx = body2idx(bodies, head_rdict)
            if torch.cuda.is_available():
                inputs = torch.LongTensor(np.array(body_idx)).to(device)
            else:
                inputs = torch.LongTensor(np.array(body_idx))

            print("## body {}".format((epoches+1)* batch_size))

            with torch.no_grad():
                pred_head, _entropy_loss = model(inputs) # [batch_size, 2*n_rel+1]
                prob_ = torch.softmax(pred_head, dim=-1)
                probs.append(prob_.detach().cpu())

        rule_conf[rule_len] = torch.cat(probs,dim=0)
        print ("rule_conf",rule_conf[rule_len].shape)

if __name__ == '__main__':
    msg = "First Order Logic Rule Mining"
    print_msg(msg)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="increase output verbosity")
    parser.add_argument("--test", action="store_true", help="increase output verbosity")
    parser.add_argument("--get_rule", action="store_true", help="increase output verbosity")
    parser.add_argument("--data", default="family", help="increase output verbosity")
    parser.add_argument("--topk", type=int, default=200, help="increase output verbosity")
    parser.add_argument("--anchor", type=int, default=10000, help="increase output verbosity")
    parser.add_argument("--gpu", type=int, default=1, help="increase output verbosity")
    parser.add_argument("--output_file", default="ours_family", help="increase output verbosity")
    parser.add_argument("--model", default="family", help="increase output verbosity")
    parser.add_argument("--max_path_len", type=int, default=3, help="increase output verbosity")
    parser.add_argument("--learned_path_len", type=int, default=2, help="increase output verbosity")
    parser.add_argument("--sparsity", type=float, default=1, help="increase output verbosity")
    parser.add_argument("--alpha", type=float, default=1, help="increase output verbosity")
    parser.add_argument("--batch_size", type=int, default=1000, help="increase output verbosity")
    parser.add_argument("--n_epoch", type=int, default=1500, help="increase output verbosity")


    args = parser.parse_args()
    assert args.train or args.test

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # DataSet
    data_path = '../datasets/{}/'.format(args.data)
    dataset = Dataset(data_root=data_path, sparsity=args.sparsity, inv=True)
    print("Dataset:{}".format(data_path))

    
    # Saved Model
    model_path = '../results/{}'.format(args.model)
    print("Model:{}".format(model_path))

    if args.train:
        print_msg("Train!")
        train(args, dataset)

    if args.test:
        print_msg("Test!")
        test(args, dataset)
            
        if args.get_rule:
            print_msg("Generate Rule!")
            
            head_rdict = dataset.get_head_relation_dict()
            n_rel = head_rdict.__len__()-1
            # TODO 保存文件改一下
            for rule_len in rule_conf:
                # rule_path = "./{}_{}_{}.txt".format(args.output_file, args.topk, rule_len)
                rule_path = "./{}/{}_{}_{}.txt".format(args.output_file, args.output_file, args.topk, rule_len)
                print("\nrule length:{}".format(rule_len))
                sorted_val, sorted_idx = torch.sort(rule_conf[rule_len],0, descending=True)

                n_rules, _ = sorted_val.shape

                with open(rule_path, 'w') as g:
                    for r in range(n_rel):
                        head = head_rdict.idx2rel[r]
                        idx = 0
                        while idx<args.topk and idx<n_rules:
                            conf = sorted_val[idx, r]
                            body = candidate_rule[rule_len][sorted_idx[idx, r]]
                            msg = "{:.3f} ({:.3f})\t{} <-- ".format(conf, conf, head)
                            body = body.split('|')
                            msg += ", ".join(body)
                            g.write(msg + '\n')
                            idx+=1

