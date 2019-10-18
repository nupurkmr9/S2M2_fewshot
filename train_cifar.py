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




def train_manifold_mixup(base_loader, base_loader_test, model, start_epoch, stop_epoch, params):

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    print("stop_epoch"  , start_epoch, stop_epoch)

    for epoch in range(start_epoch, stop_epoch):
        print('\nEpoch: %d' % epoch)

        model.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        correct1 = 0.0
        total = 0

        for batch_idx, (input_var, target_var) in enumerate(trainloader):
            input_var, target_var = input_var.cuda(), target_var.cuda()
            input_var, target_var = Variable(input_var), Variable(target_var)
            lam = np.random.beta(args.alpha, args.alpha)
            _ , outputs , target_a , target_b  = model(input_var, target_var, mixup_hidden= True, mixup_alpha = args.alpha , lam = lam)
            loss = mixup_criterion(criterion, outputs, target_a, target_b, lam)
            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target_var.size(0)
            correct += (lam * predicted.eq(target_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(target_b.data).cpu().sum().float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx%50 ==0 :
                print('{0}/{1}'.format(batch_idx,len(trainloader)), 'Loss: %.3f | Acc: %.3f%%  | Orig Acc:  %.3f%% '
                             % (train_loss/(batch_idx+1),100.*correct/total , 100.*correct/total))
        

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict() }, outfile)
         

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(base_loader_test):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            f , outputs = model.forward(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print('Loss: %.3f | Acc: %.3f%%'
                         % (test_loss/(batch_idx+1), 100.*correct/total ))


        
    return model 




def train_rotation(base_loader, base_loader_test, model, optimization, start_epoch, stop_epoch, params , tmp):
    rotate_classifier = nn.Sequential( nn.Linear(640,4)) 
    rotate_classifier.cuda()
    
    if 'rotate' in tmp:
        print("loading rotate model")
        rotate_classifier.load_state_dict(tmp['rotate'])
        
    optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': rotate_classifier.parameters()}
            ])
    
    lossfn = nn.CrossEntropyLoss()
    max_acc = 0 

    print("stop_epoch" , start_epoch, stop_epoch )

    for epoch in range(start_epoch,stop_epoch):
        rotate_classifier.train()
        model.train()

        avg_loss=0
        avg_rloss=0
        
        for i, (x,y) in enumerate(base_loader):
            bs = x.size(0)
            x_ = []
            y_ = []
            a_ = []
            for j in range(bs):
                x90 = x[j].transpose(2,1).flip(1)
                x180 = x90.transpose(2,1).flip(1)
                x270 =  x180.transpose(2,1).flip(1)
                x_ += [x[j], x90, x180, x270]
                y_ += [y[j] for _ in range(4)]
                a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

            x_ = Variable(torch.stack(x_,0)).cuda()
            y_ = Variable(torch.stack(y_,0)).cuda()
            a_ = Variable(torch.stack(a_,0)).cuda()
            f,scores = model.forward(x_)
            rotate_scores =  rotate_classifier(f)

            optimizer.zero_grad()
            rloss = lossfn(rotate_scores,a_)
            closs = lossfn(scores, y_)
            loss = 0.5*closs + 0.5*rloss
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss+closs.data.item()
            avg_rloss = avg_rloss+rloss.data.item()
            

            if i % 50 ==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Rotate Loss {:f}'.format(epoch, i, len(base_loader), avg_loss/float(i+1),avg_rloss/float(i+1)  ))
            
            
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict() , 'rotate': rotate_classifier.state_dict()}, outfile)
         

                
        model.eval()
        rotate_classifier.eval()
        correct = rcorrect = total = 0
        for i,(x,y) in enumerate(base_loader_test):
            if i<2:
                bs = x.size(0)
                x_ = []
                y_ = []
                a_ = []
                for j in range(bs):
                    x90 = x[j].transpose(2,1).flip(1)
                    x180 = x90.transpose(2,1).flip(1)
                    x270 =  x180.transpose(2,1).flip(1)
                    x_ += [x[j], x90, x180, x270]
                    y_ += [y[j] for _ in range(4)]
                    a_ += [torch.tensor(0),torch.tensor(1),torch.tensor(2),torch.tensor(3)]

                x_ = Variable(torch.stack(x_,0)).cuda()
                y_ = Variable(torch.stack(y_,0)).cuda()
                a_ = Variable(torch.stack(a_,0)).cuda()
                f,scores = model(x_)
                rotate_scores =  rotate_classifier(f)
                p1 = torch.argmax(scores,1)
                correct += (p1==y_).sum().item()
                total += p1.size(0)
                p2 = torch.argmax(rotate_scores,1)
                rcorrect += (p2==a_).sum().item()

        print("Epoch {0} : Accuracy {1}, Rotate Accuracy {2}".format(epoch,(float(correct)*100)/total,(float(rcorrect)*100)/total))
        

    return model


if __name__ == '__main__':
    params = parse_args('save_features')
    
    image_size = 32


    split = params.split
    base_file = configs.data_dir[params.dataset] + 'base.json'
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    if params.save_iter != -1:
        modelfile   = get_assigned_file(checkpoint_dir,params.save_iter)
    else:
        modelfile   = get_resume_file(checkpoint_dir)
    

    base_datamgr    = SimpleDataManager(image_size, batch_size = params.batch_size)
    base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
    val_datamgr    = SimpleDataManager(image_size, batch_size = 32)
    val_loader     = base_datamgr.get_data_loader( base_file , aug = False )


    if params.method == 'manifold_mixup':
        model = wrn_mixup_model.wrn28_10(64)
    elif params.method == 'S2M2_R':
        model = wrn_mixup_model.wrn28_10(64)
    elif params.method == 'rotation':
        model = BaselineTrain( model_dict[params.model], 64, loss_type = 'dist')
 
            
    
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
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state[key.replace("classifier.","linear.")] =  state[key]
                state.pop(key)

        
        model.load_state_dict(state)        
        model = torch.nn.DataParallel(model, [0,1,2])  
        model.cuda()
        model = train_manifold_mixup(base_loader, val_loader, model, start_epoch, start_epoch+stop_epoch, params)


    elif params.method =='rotation':
        model = train_rotation(base_loader, val_loader, model, start_epoch, stop_epoch, params,tmp):

    elif params.method == 'manifold_mixup':
        model = train_manifold_mixup(base_loader, val_loader, model, start_epoch, stop_epoch, params):



  
