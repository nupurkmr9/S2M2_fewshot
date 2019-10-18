#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.datamgr import SimpleDataManager
# import models
# from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
import wrn_mixup_model
from io_utils import model_dict, parse_args, get_resume_file ,get_assigned_file


parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')


use_cuda = torch.cuda.is_available()


if __name__ == '__main__':
    params = parse_args('save_features')
    
    image_size = 80


    split = params.split
    base_file = configs.data_dir[params.dataset] + 'base.json'
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    

    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
    else:
        modelfile   = get_resume_file(checkpoint_dir)
    

    base_datamgr    = SimpleDataManager(image_size, batch_size = params.batch_size)
    base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
    val_datamgr    = SimpleDataManager(image_size, batch_size = 32)
    val_loader     = base_datamgr.get_data_loader( base_file , aug = False )


    if params.method == 'manifold_mixup':
        model = wrn_mixup_model.wrn28_10(200)
    elif params.method == 'S2M2_R':
        model = wrn_mixup_model.wrn28_10(200)
    else:
        model = model_dict[params.model]()

 
            
if use_cuda:
    model = torch.nn.DataParallel(model, [0,1,2,3,4,5,6,7])   
    
    if params.method =='S2M2_R'
        resume_rotate_file_dir = checkpoint_dir.replace("S2M2_R","rotation")
        resume_file = get_resume_file( resume_rotate_file_dir )
        
        print("resume_file" , resume_file)
        tmp = torch.load(resume_file)
        start_epoch = tmp['epoch']+1
        print("restored epoch is" , tmp['epoch'])
        state = tmp['state']
        state_keys = list(state.keys())
        for i, key in enumerate(state_keys):
            if 'linear' in key:
                state.pop[key]
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state.pop(key)
        
        model_dict_load = model.state_dict()
        model_dict_load.update(state)
        model.load_state_dict(model_dict_load)
        
        
    
    model.cuda()
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()

bce_loss = nn.BCELoss().cuda()

softmax = nn.Softmax(dim=1).cuda()


# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov = True, weight_decay=args.decay)

# def lr_lambda(step):
#     if step in range(0,200):
#         return 1.0
#     elif step in range(200,300):
#         return 0.1
#     elif step >=300:
#         return 0.1
#     else:
#         return 1
    
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov = True, weight_decay=args.decay)

# def lr_lambda(step):
#     if step in range(0,60):
#         return 1.0
#     elif step in range(60,120):
#         return 0.1
#     elif step in range(120,180):
#         return 0.1
#     elif step >=180:
#         return 1.0
#     else:
#         return 1
    
    
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


optimizer = optim.Adam(model.parameters(), lr=args.lr)

def lr_lambda(step):
    if step in range(0,300):
        return 1.0
    else:
        return 1
    
    
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    correct1 = 0.0
    total = 0
    for batch_idx, (input_var, target_var) in enumerate(trainloader):
        if use_cuda:
            input_var, target_var = input_var.cuda(), target_var.cuda()

        input_var, target_var = Variable(input_var), Variable(target_var)
        lam = np.random.beta(args.alpha, args.alpha)
        outputs , target_a , target_b , reweighted_target = model(input_var, target_var, mixup_hidden= True, mixup_alpha = args.alpha , lam = lam)
#         print(softmax(outputs) , reweighted_target.nonzero())
#         loss = bce_loss(softmax(outputs), reweighted_target)
        loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += target_var.size(0)
        correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
#         _ , outputs_in =  model.forward(input_var)
#         _, predicted_in = torch.max(outputs_in.data, 1)
#         correct1 += predicted_in.eq(target_var.data).cpu().sum().float()
        
        if batch_idx%50 ==0 :
            print('{0}/{1}'.format(batch_idx,len(trainloader)), 
                         'Loss: %.3f | Acc: %.3f%%  | Orig Acc:  %.3f%%  '
                         % (train_loss/(batch_idx+1),
                            100.*correct/total , 100.*correct/total))
    
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        f , outputs = model.forward(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('Loss: %.3f | Acc: %.3f%%'
                     % (test_loss/(batch_idx+1), 100.*correct/total,
                        correct, total))
    acc = 100.*correct/total
    if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        checkpoint(acc, epoch)
    if acc > best_acc:
        best_acc = acc
    return (test_loss/batch_idx, 100.*correct/total)
       
def checkpoint(epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'epoch': epoch,
        'state': model.state_dict()
    }
    checkpoint_file = args.model + '_baseline_aug_manifold_mixup_overexemplars4l_drop5'
    if not os.path.isdir('checkpoints/miniImagenet/' + checkpoint_file):
        os.mkdir('checkpoints/miniImagenet/' + checkpoint_file)
    torch.save(state, './checkpoints/miniImagenet/{0}/{1}.tar'.format(checkpoint_file,epoch))

    
val_best_acc = 0.0
loader = torch.load('./imagenet_val.pt')

for epoch in range(start_epoch, args.epoch):
    train_loss, reg_loss, acc = train(epoch)
    scheduler.step()
    if (epoch%5==0 or epoch == args.epoch - 1):
        checkpoint(epoch)
   
    valmodel =  BaselineFinetune(model_dict['WideResNet28_10'],5,1,loss_type='dist')
    valmodel.n_query=15
    acc_all1, acc_all2 , acc_all3 = [],[],[]
    for i,x in enumerate(loader):
        x = x.view(-1,3,80,80)
#         print(x.size())
        f , _ = model(x.cuda())
        f = f.view(5,16,-1)
#         print(f.size())
        scores  = valmodel.set_forward_adaptation(f.cpu())
        acc = []
#         print(len(scores))
#         print(scores[0])
        for each_score in scores:
#             print(each_score.size())
            pred = each_score.data.cpu().numpy().argmax(axis = 1)
            y = np.repeat(range( 5 ), 15 )
            acc.append(np.mean(pred == y)*100 )
        acc_all1.append(acc[0])
        acc_all2.append(acc[1])
        acc_all3.append(acc[2])

    print('Test Acc at 100= %4.2f%%' %(np.mean(acc_all1)))
    print('Test Acc at 200= %4.2f%%' %(np.mean(acc_all2)))
    print('Test Acc at 300= %4.2f%%' %(np.mean(acc_all3)))

    if np.mean(acc_all3) > val_best_acc:
        val_best_acc = np.mean(acc_all3)
        checkpoint_file = args.model + '_baseline_aug_manifold_mixup_overexemplars4l_drop5'
        bestfile =os.path.join('checkpoints/miniImagenet/' + checkpoint_file, 'best.tar')
        torch.save({'epoch':epoch, 'state':model.state_dict()}, bestfile)

