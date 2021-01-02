import argparse
import os
import time
import numpy as np

import shutil
import ntpath
import sys
import pdb
import logging
import gc
#import resource

from importlib import import_module
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from data_utils.torch_data import THOR_Data, get_cross_validation_paths, get_global_alpha
import data_utils.transforms as tr
from utils import setgpu, get_threshold, metric, segmentation_metrics
from torchsummary import summary


###########################################################################
"""
                The main function of SegTHOR
                      Python 3
                    pytorch 1.1.0
                   author: Tao He
              Institution: Sichuan University
               email: taohe@stu.scu.edu.cn
"""
###########################################################################

parser = argparse.ArgumentParser(description='PyTorch SegTHOR Segmentation')
parser.add_argument(
    '--model_name',
    '-m_name',
    metavar='MODEL',
    default='ResUNet101_lmser',
    help='model_name')
parser.add_argument(
    '--normal_epochs',
    default=40,
    type=int,
    metavar='N',
    help='number of max normal epochs to run')
parser.add_argument(
    '--lmser_epochs',
    default=40,
    type=int,
    metavar='N',
    help='number of max lmser epochs to run')
parser.add_argument(
    '-b',
    '--batch-size',
    default=4,
    type=int,
    metavar='N',
    help='mini-batch size (default: 16)')
parser.add_argument(
    '--lr',
    '--learning-rate',
    default=0.01,
    type=float,
    metavar='LR',
    help='initial learning rate')
parser.add_argument(
    '--momentum', 
    default=0.9, 
    type=float, 
    metavar='M', 
    help='momentum')
parser.add_argument(
    '--weight-decay',
    '--wd',
    default=0.00001,
    type=float,
    metavar='W',
    help='weight decay (default: 1e-5)')
parser.add_argument(
    '--save_dir',
    default='SavePath/ResUNet101_lmser/',
    type=str,
    metavar='SAVE',
    help='directory to save checkpoint (default: none)')
parser.add_argument(
    '--gpu', 
    default='0', 
    type=str, 
    metavar='N', 
    help='use gpu')
parser.add_argument(
    '--patient',
    default=10,
    type=int,
    metavar='N',
    help='the flat to stop training')
parser.add_argument(
    '--untest_epoch',
    default=10,
    type=int,
    metavar='N',
    help='number of untest_epoch, do not test for n epoch. just for saving time')
parser.add_argument(
    '--loss_name',
    default='CombinedLoss',
    type=str,
    metavar='N',
    help='the name of loss function')
parser.add_argument(
    '--data_path',
    default='/home/data/mahaoran/lei_nao_data/data_npy',
    type=str,
    metavar='N',
    help='data path')
parser.add_argument(
    '--test_flag',
    default=0,
    type=int,
    metavar='0, 1, 2, 3',
    help='the test flag range in 0..9, 10..19, 20..29, 30..39 !')
parser.add_argument(
    '--n_class',
    default=5,
    type=int,
    metavar='n_class',
    help='number of classes')

DEVICE = torch.device("cuda" if True else "cpu")


def main(args):
    torch.manual_seed(123)
    cudnn.benchmark = True
    setgpu(args.gpu)
    data_path = args.data_path
    train_files, test_files = get_cross_validation_paths(args.test_flag) 

    model = import_module('models.model_loader')

    net_dict, loss = model.get_full_model(
        args.model_name, 
        args.loss_name, 
        n_classes=args.n_class)

    save_dir = args.save_dir
    logging.info(args)

    for net_name, net in net_dict.items():
        net = net.to(DEVICE)
    #summary(net, (3, 512, 512))
    loss = loss.to(DEVICE)

    optimizer_dict = {}
    for net_name, net in net_dict.items():
        optimizer_dict[net_name] = torch.optim.SGD(
                                        net.parameters(),
                                        args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        '''
        for name, param in net.named_parameters():
            if param.requires_grad:
                print(name)
                print(param.shape)
        '''
    init_lr = np.copy(args.lr)
    def get_lr(epoch):
        if args.lr < 0.0001:
            return args.lr
        if epoch > 0:
            args.lr = args.lr * 0.95
            logging.info('current learning rate is %f' % args.lr)
        return args.lr
    
    composed_transforms_tr = transforms.Compose([
    tr.RandomZoom((512, 512)),
    tr.RandomHorizontalFlip(),
    tr.Normalize(mean=(0.12, 0.12, 0.12), std=(0.018, 0.018, 0.018)),
    tr.ToTensor2(args.n_class)])
    train_dataset = THOR_Data(
        transform=composed_transforms_tr, path=data_path, file_list=train_files)
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4)


    # Pretrain ResUNet101
    for epoch_normal in range(args.normal_epochs):
        train_normal(trainloader, net_dict, loss, epoch_normal, optimizer_dict, get_lr, save_dir)
        print('Pretrain ResUNet101 epoch %d done.' % epoch_normal)
        
        # Save state
        net_state_dict = []
        for net_name, net in net_dict.items():
        	net_state_dict.append(net.state_dict())
        optimizer_state_dict = []
        for net_name, optimizer in optimizer_dict.items():
        	optimizer_state_dict.append(optimizer.state_dict())

        torch.save({
            'epoch': epoch_normal,
            'save_dir': save_dir,
            'state_dict': net_state_dict,
            'optimizer': optimizer_state_dict,
            'args': args
        }, os.path.join(save_dir, '%d.ckpt' % epoch_normal))
 
    # Train ResUNet101_lmser based on ResUNet101
    for epoch_lmser in range(args.lmser_epochs):
        train_lmser(trainloader, net_dict, loss, epoch_lmser, optimizer_dict, get_lr, save_dir)
        print('Train ResUNet101_lmser epoch %d done.' % epoch_lmser)



def train_normal(data_loader, net_dict, loss, epoch, optimizer_dict, get_lr, save_dir):
    for net_name, net in net_dict.items():
        net.train()
    lr = get_lr(epoch)
    for net_name, optimizer in optimizer_dict.items():
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
    for i, sample in enumerate(data_loader):
        data = sample['image']
        target_c = sample['label_c']
        target_s = sample['label_s']
        data = data.to(DEVICE)
        target_c = target_c.to(DEVICE)
        target_s = target_s.to(DEVICE)

        # ResUnet
        x3 = net_dict['net_down_3'](data)
        x2 = net_dict['net_down_2'](x3)
        x1 = net_dict['net_down_1'](x2)
        out = net_dict['net_down_0'](x1)
        y1 = net_dict['net_up_0'](out, x1)
        y2 = net_dict['net_up_1'](y1, x2)
        y3 = net_dict['net_up_2'](y2, x3)
        output_s, output_c = net_dict['net_up_3'](y3)

        # All operations from down3 to up3 are stored in the gradient of the variable
        cur_loss, _, _, c_p = loss(output_s, output_c, target_s, target_c)
        cur_loss.backward() # Backward for all steps from down3 to up3

        # Each network update parameters separately
        for net_name, optimizer in optimizer_dict.items():
            optimizer.zero_grad()
            optimizer.step()
        


def train_lmser(data_loader, net_dict, loss, epoch, optimizer_dict, get_lr, save_dir):
    for net_name, net in net_dict.items():
        net.train()
    lr = get_lr(epoch)
    for net_name, optimizer in optimizer_dict.items():
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    old_state_dict = {}
    for net_name, optimizer in optimizer_dict.items(): 
        old_state_dict[net_name] = optimizer.state_dict()

    # Train layer 0 
    train_net_name_0 = ['net_down_0', 'net_up_0']
    convergence_threshold_0 = 1
    train_lmser_layer(train_net_name_0, data_loader, net_dict, loss, optimizer_dict, convergence_threshold_0, old_state_dict)
    print('Train ResUNet101_lmser epoch %d layer 0 done.' % epoch)

    # Train layer 0 1
    train_net_name_01 = ['net_down_1', 'net_down_0', 'net_up_0', 'net_up_1']
    convergence_threshold_01 = 1
    train_lmser_layer(train_net_name_01, data_loader, net_dict, loss, optimizer_dict, convergence_threshold_01, old_state_dict)
    print('Train ResUNet101_lmser epoch %d layer 0 1 done.' % epoch)

    # Train layer 0 1 2
    train_net_name_012 = ['net_down_2', 'net_down_1', 'net_down_0', 'net_up_0', 'net_up_1', 'net_up_2']
    convergence_threshold_012 = 1
    train_lmser_layer(train_net_name_012, data_loader, net_dict, loss, optimizer_dict, convergence_threshold_012, old_state_dict)
    print('Train ResUNet101_lmser epoch %d layer 0 1 2 done.' % epoch)

    # Train layer 0 1 2 3
    train_net_name_0123 = ['net_down_3', 'net_down_2', 'net_down_1', 'net_down_0', 'net_up_0', 'net_up_1', 'net_up_2', 'net_up_3']
    convergence_threshold_0123 = 1
    train_lmser_layer(train_net_name_0123, data_loader, net_dict, loss, optimizer_dict, convergence_threshold_0123, old_state_dict)
    print('Train ResUNet101_lmser epoch %d layer 0 1 2 3 done.' % epoch)

def cycleone(x2,y1):
    x1 = (net_dict['net_down_1'](x2) + y1) / 2
    out = net_dict['net_down_0'](x1)
    y1 = net_dict['net_up_0'](out, x1)
    return x1,y1

def cycletwo(x3,y2):
    x2 = (net_dict['net_down_2'](x3) + y2) / 2
    x1 = net_dict['net_down_1'](x2)
    out = net_dict['net_down_0'](x1)
    y1 = net_dict['net_up_0'](out, x1)
    y2 = net_dict['net_up_1'](y1, x2)
    return x2,y2

def cyclethree(data,y3):
    x3 = (net_dict['net_down_3'](data)+y3)/2
    x2 = net_dict['net_down_2'](x3)
    x1 = net_dict['net_down_1'](x2)
    out = net_dict['net_down_0'](x1)
    y1 = net_dict['net_up_0'](out, x1)
    y2 = net_dict['net_up_1'](y1, x2)
    y3 = net_dict['net_up_2'](y2, x3)
    return x3,y3


def train_lmser_layer(train_net_name, data_loader, net_dict, loss, optimizer_dict,
                      convergence_threshold, old_state_dict,loss_mse = torch.nn.MSELoss(reduce=False, size_average=False)):
    while True:
        for i, sample in enumerate(data_loader):
            data = sample['image']
            target_c = sample['label_c']
            target_s = sample['label_s']
            data = data.to(DEVICE)
            target_c = target_c.to(DEVICE)
            target_s = target_s.to(DEVICE)

            # ResUnet
            '''
            x3 = net_dict['net_down_3'](data) 
            x2 = net_dict['net_down_2'](x3) 
            x1 = net_dict['net_down_1'](x2) 
            out = net_dict['net_down_0'](x1)
            y1 = net_dict['net_up_0'](out, x1) 
            y2 = net_dict['net_up_1'](y1, x2)  
            y3 = net_dict['net_up_2'](y2, x3)  
            output_s, output_c = net_dict['net_up_3'](y3) 
            '''
            x3 = net_dict['net_down_3'](data)
            x2 = net_dict['net_down_2'](x3)
            x1 = net_dict['net_down_1'](x2)
            out = net_dict['net_down_0'](x1)
            y1 = net_dict['net_up_0'](out, x1)
            old_y1 = y1
            x1,y1 = cycleone(x2,y1)
            while not break_condition(y1,old_y1):
                #continue
                cycleoneloss = loss_mse(y1,old_y1)
                cycleoneloss.backward()
                #for optimizer in [optimizer_dict['net_up_0'],optimizer_dict['net_down_0'],optimizer_dict['net_down_1']]:
                for optimizer in optimizer_dict[3:5]:
                    optimizer.zero_grad()
                    optimizer.step()
                old_y1 = y1
                x1,y1 = cycleone(x2,y1)


            y2 = net_dict['net_up_1'](y1, x2)
            old_y2 = y2
            x2,y2 = cycleone(x3,y2)
            while not break_condition(y2,old_y2):
                cycletwoloss = loss_mse(y2,old_y2)
                cycletwoloss.backward()

                for optimizer in optimizer_dict[2:6]:
                    optimizer.zero_grad()
                    optimizer.step()
                old_y2 = y2
                x2, y2 = cycleone(x3, y2)

            y3 = net_dict['net_up_2'](y2, x3)
            old_y3 = y3
            x3,y3 = cyclethree(data,y3)
            while not break_condition(y3, old_y3):
                cyclethreeloss = loss_mse(y3, old_y3)
                cyclethreeloss.backward()
                for optimizer in optimizer_dict[1:7]:
                    optimizer.zero_grad()
                    optimizer.step()
                old_y3 = y3
                x3, y3 = cyclethree(data, y3)



            output_s, output_c = net_dict['net_up_3'](y3)

            # All operations from down3 to up3 are stored in the gradient of the variable
            cur_loss, _, _, c_p = loss(output_s, output_c, target_s, target_c)
            cur_loss.backward() # Backward for all steps from down3 to up3

            # update parameters for certain layer  
            for net_name, optimizer in optimizer_dict.items():
                if net_name not in train_net_name:
                    continue
                optimizer.zero_grad()
                optimizer.step()

            # Break condition
            break_flag = True
            for net_name, optimizer in optimizer_dict.items():
                if net_name not in train_net_name:
                    continue
                print(net_dict[net_name].state_dict()['down0.layer1.15.conv1.weight'])

                #print(optimizer.param_groups[0]['params'][0])
                if not break_condition(optimizer.state_dict(), old_state_dict[net_name], convergence_threshold):
                    break_flag = False
                    break

            if break_flag:
                return
            else:
                for net_name, optimizer in optimizer_dict.items(): 
                    old_state_dict['net_name'] = optimizer.state_dict()

def break_condition(data,old_data):
    error = data - old_data
    if torch.sum(torch.abs(error)) < torch.sum(torch.abs(old_data))/10:
        return True
    else:
        return False

'''
def break_condition(state_dict, old_state_dict, threshold):
    parameter_loss = np.zeros((len(state_dict.items())))
    i = 0
    for p_name, p in state_dict.items():
        parameter_loss[i] = p - old_state_dict[p_name]
        i += 1
    print(parameter_loss.sum(axis=0))
    if parameter_loss.sum(axis=0) <= convergence_threshold:
        return True
    else:
        return False
'''


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, args.model_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, str(args.test_flag))
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s,%(lineno)d: %(message)s\n',
        datefmt='%Y-%m-%d(%a)%H:%M:%S',
        filename=os.path.join(args.save_dir, 'log.txt'),
        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    main(args)
    
    
    
