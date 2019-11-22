import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file
import random

def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list,n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
   
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores  = model.set_forward(z_all, is_feature = True)
    
    acc = []
    for each_score in scores:
        pred = each_score.data.cpu().numpy().argmax(axis = 1)
        y = np.repeat(range( n_way ), n_query )
        acc.append(np.mean(pred == y)*100 )
    return acc



if __name__ == '__main__':
    params = parse_args('test')

    acc_all = []

    if params.dataset == 'CUB':
        iter_num = 600
    else:
        iter_num = 10000

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 


    model = BaselineFinetune( model_dict[params.model], **few_shot_params )

    if torch.cuda.is_available():
        model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)


    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split

    novel_file = os.path.join( checkpoint_dir.replace("checkpoints","features"), split_str +".hdf5") 
    cl_data_file = feat_loader.init_loader(novel_file)
        
    acc_all1, acc_all2 , acc_all3 = [],[],[]

    if params.dataset == 'CUB':
        n_query = 15
    else:
        n_query = 600 - params.n_shot

    print(novel_file)
    print("evaluating over %d examples"%(n_query))

    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, model, n_query = n_query , adaptation = params.adaptation, **few_shot_params)
            
        acc_all1.append(acc[0])
        acc_all2.append(acc[1])
        acc_all3.append(acc[2])
        print("%d steps reached and the mean acc is %g , %g , %g"%(i, np.mean(np.array(acc_all1)),np.mean(np.array(acc_all2)),np.mean(np.array(acc_all3)) ))
#         acc_all  = np.asarray(acc_all)
    acc_mean1 = np.mean(acc_all1)
    acc_mean2 = np.mean(acc_all2)
    acc_mean3 = np.mean(acc_all3)
    acc_std1  = np.std(acc_all1)
    acc_std2  = np.std(acc_all2)
    acc_std3  = np.std(acc_all3)
    print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' %(iter_num, acc_mean1, 1.96* acc_std1/np.sqrt(iter_num)))
    print('%d Test Acc at 200= %4.2f%% +- %4.2f%%' %(iter_num, acc_mean2, 1.96* acc_std2/np.sqrt(iter_num)))
    print('%d Test Acc at 300= %4.2f%% +- %4.2f%%' %(iter_num, acc_mean3, 1.96* acc_std3/np.sqrt(iter_num)))
