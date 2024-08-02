# -*- coding: utf-8 -*-
# JM

import datetime
start_time = datetime.datetime.now()

import os, sys, re
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import timm
from lib_model.dataset import BasicDataset

mse_metric = nn.MSELoss()

def loss_mae(y_target, y_pred):
    return torch.sum(torch.abs(y_target - y_pred)) / y_target.size(0)

def loss_mse(y_target, y_pred):
    return torch.sum(torch.square(y_target - y_pred)) / y_target.size(0)

def loss_rmse(y_target, y_pred):
    return torch.sqrt(torch.sum(torch.square(y_target - y_pred)) / y_target.size(0))

def r_squre(y_target, y_pred):
    mean = torch.mean(y_target)
    ss_total = torch.sum(torch.square(y_target - mean))
    ss_fit = torch.sum(torch.square(y_target - y_pred))
    r_2 = 1 - ss_fit/ss_total
    return r_2


def train_net(net, device, epochs, name_model, batch_size, patch_size, patch_num, lr, step_size, gamma, path_data_train, path_data_val):
    time_stamp = '-'.join(re.split('[:|\s+]+',str(datetime.datetime.now())[:16]))
    time_stamp = time_stamp[:4] + time_stamp[5:7] + time_stamp[8:11] + time_stamp[11:13] + time_stamp[14:16]

    name_metric = 'R2'
    accuracy_best = 0.5
    accuracy_save = 0.7
    
    train_data = BasicDataset(path_data_train, patch_size = patch_size, patch_num = patch_num, mode = 'train')
    val_data = BasicDataset(path_data_val, patch_size = patch_size, patch_num=patch_num, mode = 'val')
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, drop_last = True)
    
    name_case = f"{name_model}_Epoch_{epochs}_BS_{batch_size}_LR_{lr}_PS_{patch_size}_PN_{patch_num}_{time_stamp}"
    path_summary = f"output/{name_case}/summary" 
    os.makedirs(path_summary, exist_ok=True)
    writer = SummaryWriter(path_summary)
    
    print(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Patch size:      {patch_size} x {patch_size}
            Patches/image:   {patch_num}
            Training size:   {len(train_data)}
            Validation size: {len(val_data)}
            Device:          {device.type}''')
        
    optimizer = optim.Adam(net.parameters(), lr = lr, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
    
    for epoch in range(epochs):
        start_time_epoch = datetime.datetime.now()
        
        net.train()
        
        loss_train_total = []
        with tqdm(total=len(train_data), desc=f'Epoch {epoch+1}/{epochs}', unit=' img') as prog_bar:
            for iteration, batch_train in enumerate(train_loader):
                input_batch_train = batch_train['image']
                gt_batch_train = batch_train['label']
                
                for i in range(patch_num):
                    image_train = input_batch_train[:, (i*3): 3+(i*3), :, :].to(device=device, dtype=torch.float32)
                    gt_batch_train = gt_batch_train.to(device=device, dtype=torch.float32)
                    
                    lnc_pred_train = net(image_train)
                    
                    loss_train = mse_metric(gt_batch_train, lnc_pred_train)
                    loss_train_total.append(loss_train.item())
                    
                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()
                    
                    prog_bar.set_postfix(**{'loss (batch)': loss_train.item()})
                    prog_bar.update(np.ceil(image_train.shape[0]/patch_num))
        
        loss_train_mean = np.mean(np.array(loss_train_total))
        
        ## Validation ----------------------------------------------------------------------------------------------------------------------------
        net.eval()
        
        loss_val_total = []
        lnc_gt_val_ens = torch.zeros(0, 1).to(device = device, dtype = torch.float32)
        lnc_pred_val_ens = torch.zeros(0, 1).to(device = device, dtype = torch.float32)
        
        with tqdm(total=len(val_loader), desc='Validation round', unit=' batch') as prog_bar:
            for iteration, batch_val in enumerate(val_loader):
                input_batch_val = batch_val['image']
                gt_batch_val = batch_val['label']
                
                image_val = input_batch_val[:, 0:3, :, :].to(device = device, dtype = torch.float32)
                gt_batch_val = gt_batch_val.to(device = device, dtype = torch.float32)
                
                with torch.no_grad():
                    lnc_pred_val = net(image_val)
                    
                    loss_val = mse_metric(gt_batch_val, lnc_pred_val)
                    loss_val_total.append(loss_val.item())
                    
                lnc_gt_val_ens = torch.cat((lnc_gt_val_ens, gt_batch_val), dim=0)
                lnc_pred_val_ens = torch.cat((lnc_pred_val_ens, lnc_pred_val), dim=0)
                
                prog_bar.update(np.ceil(image_val.shape[0]/patch_num))
        
        loss_val_mean = np.mean(np.array(loss_val_total))
        r2_val = r_squre(lnc_gt_val_ens, lnc_pred_val_ens).cpu().numpy()
        rmse_val = loss_rmse(lnc_gt_val_ens, lnc_pred_val_ens).cpu().numpy()
        
        
        if r2_val > accuracy_best:
            accuracy_best = r2_val
            if accuracy_best > accuracy_save:
                path_model = f"output/{name_case}/model"
                
                os.makedirs(path_model, exist_ok=True)
                torch.save(net.state_dict(), path_model + 
                           f"/{name_model}_{name_metric}_{'%.3f'%accuracy_best}_Epoch_{'%03d'%(epoch+1)}_BS_{batch_size}_LR_{lr}_PS_{patch_size}_PN_{patch_num}.pth")
                
                print(f"Save the best model: R²={'%.3f'%accuracy_best}")
        
        print(f"Validation R²: {'%.3f'%r2_val} / {'%.3f'%accuracy_best}")
        print(f"Validation RMSE: {'%.3f'%rmse_val}, Train MSE: {'%.3f'%loss_train_mean}, Val MSE: {'%.3f'%loss_val_mean}")
        
        writer.add_scalar('loss/train', loss_train_mean, epoch+1)
        writer.add_scalar('loss/val', loss_val_mean, epoch+1)
        writer.add_scalar('val/r2', r2_val, epoch+1)
        writer.add_scalar('val/rmse', rmse_val, epoch+1)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch+1)
        
        scheduler.step()
        
        print(f"epoch time : {'%10s'%str(datetime.datetime.now()-start_time_epoch)[:7]}")
        
    writer.close()
    print('End of training')


def gpu_ens():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


if __name__ == '__main__':
    print('Training of leaf nitrogen estimation')
    # ------------------------- spliting -------------------------
    use_gpu = 0
    name_model = 'MobilenetV3'
    epochs = 500  ## 500
    batch_size = 32  ## 32
    patch_size = 224  ## 224
    patch_num = 4  ## 4
    
    lr = 0.001  ## 0.001
    step_size = 50  ## 50
    gamma = 0.5
    path_data_train = 'dataset/train'
    path_data_val = 'dataset/val'
    # ------------------------- spliting -------------------------

    if len(gpu_ens()) == 1:
        device = gpu_ens()[0]
    else:
        device = gpu_ens()[use_gpu]
    print(f'The number of GPU is {len(gpu_ens())}')
    print(f'Using device {device}')
    
    all_pretrained_models_available = timm.list_models(pretrained=False)
    net = timm.create_model('mobilenetv3_large_100', pretrained=False)
    num_classes = 1
    net.classifier = nn.Linear(1280, num_classes)
    
    net.to(device = device)
    print(f"Network parameters: {'%.2f'%(sum(param.numel() for param in net.parameters())/1e4)} w")
    
    try:
        train_net(net = net,
                  device = device,
                  epochs = epochs,
                  name_model = name_model,
                  batch_size = batch_size,
                  patch_size = patch_size,
                  patch_num = patch_num,
                  lr = lr,
                  step_size = step_size,
                  gamma = gamma,
                  path_data_train = path_data_train,
                  path_data_val = path_data_val)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'bkupCheckPoint.pth')
        print('Saved interrupt checkpoint backup')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    
    print(f"the total running time: {'%10s'%str(datetime.datetime.now() - start_time)[:7]}")
