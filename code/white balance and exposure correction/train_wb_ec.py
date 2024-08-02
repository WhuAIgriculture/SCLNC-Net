import datetime
start_time = datetime.datetime.now()

import os, sys, re
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from lib_model.dataset import BasicDataset
from lib_model.attunet import AttU_Net


def loss_mae(y, y_hat):
    return torch.sum(torch.abs(y - y_hat)) / y_hat.size(0)

def loss_mse(y, y_hat):
    return torch.sum(torch.square(y - y_hat)) / y_hat.size(0)

def loss_rmse(y, y_hat):
    return torch.sqrt(torch.sum(torch.square(y - y_hat)) / y_hat.size(0))

def r_squre(y, y_hat):
    mean = torch.mean(y)
    ss_total = torch.sum(torch.square(y - mean))
    ss_fit = torch.sum(torch.square(y - y_hat))
    r_2 = 1 - ss_fit/ss_total
    return r_2


def train_net(net, device, epochs, name_model, epoch_show, batch_size, lr, step_size, gamma, patch_size, patch_num, path_train, path_val):
    time_stamp = '-'.join(re.split('[:|\s+]+',str(datetime.datetime.now())[:16]))
    time_stamp = time_stamp[:4] + time_stamp[5:7] + time_stamp[8:11] + time_stamp[11:13] + time_stamp[14:16]
    
    name_metric = 'MAE'
    accuracy_best = 10000
    accuracy_save = 5000
    
    train_data = BasicDataset(path_train, patch_size = patch_size, patch_num = patch_num)
    val_data = BasicDataset(path_val, patch_size = patch_size, patch_num = patch_num)
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, drop_last = True)
    
    name_case = f"{name_model}_Epoch_{epochs}_BS_{batch_size}_LR_{lr}_PS_{patch_size}_PN_{patch_num}_{time_stamp}"
    path_summary = f"output/{name_case}/summary"
    os.makedirs(path_summary, exist_ok=True)
    writer = SummaryWriter(path_summary)
    
    path_visual_train = f"output/{name_case}/visual/train/"
    path_visual_val = f"output/{name_case}/visual/val/"
    os.makedirs(path_visual_train, exist_ok=True)
    os.makedirs(path_visual_val, exist_ok=True)
    
    print(f'''Starting training:
            Epochs:          {epochs} epochs
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Patch size:      {patch_size} x {patch_size}
            Patches/image:   {patch_num}
            Training size:   {len(train_data)}
            Validation size: {len(val_data)}
            Device:          {device.type}''')
        
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma = gamma)
    
    for epoch in range(epochs):
        start_time_epoch = datetime.datetime.now()
        
        net.train()
        
        loss_train_total = []
        with tqdm(total=len(train_data), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as prog_bar:
            for iteration, batch_train in enumerate(train_loader):
                input_batch_train = batch_train['image_input']
                gt_batch_train = batch_train['image_gt']
                
                for i in range(patch_num):
                    image_input_train = input_batch_train[:, (i*3): 3+(i*3), :, :].to(device=device, dtype=torch.float32)
                    image_gt_train = gt_batch_train[:, (i*3): 3+(i*3), :, :].to(device=device, dtype=torch.float32)
                    
                    image_pred_train = net(image_input_train)
                    
                    loss_train = loss_mae(image_pred_train, image_gt_train)
                    loss_train_total.append(loss_train.item())
                    
                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()
                    
                    prog_bar.set_postfix(**{'loss (batch)': loss_train.item()})
                    prog_bar.update(np.ceil(image_input_train.shape[0] / patch_num))
                
        loss_train_mean = np.mean(np.array(loss_train_total))
        
        ## Validation ----------------------------------------------------------------------------------------------------------------------------
        net.eval()
        
        loss_val_total = []
        with tqdm(total=len(val_data), desc='Validation round', unit='batch', leave=False) as prog_bar:
            for iteration, batch_val in enumerate(val_loader):
                input_batch_val = batch_val['image_input']
                gt_batch_val = batch_val['image_gt']
                
                image_input_val = input_batch_val[:, 0:3, :, :].to(device=device, dtype=torch.float32)
                image_gt_val = gt_batch_val[:, 0:3, :, :].to(device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    image_pred_val = net(image_input_val)
                    
                    loss_val = loss_mae(image_pred_val, image_gt_val)
                    loss_val_total.append(loss_val.item())
                    
                prog_bar.update(np.ceil(image_input_val.shape[0] / patch_num))
                
        loss_val_mean = np.mean(np.array(loss_val_total))
        accuracy_val = loss_val_mean
        
        ## accuracy comparison ----------------------------------------------------------------------------------------------------------------------------
        if accuracy_val < accuracy_best:
            accuracy_best = accuracy_val
            if accuracy_best < accuracy_save:
                path_model = f"output/{name_case}/model"
                os.makedirs(path_model, exist_ok=True)
                
                torch.save(net.state_dict(), path_model + 
                           f"/{name_model}_{name_metric}_{'%.0f'%accuracy_best}_Epoch_{'%03d'%(epoch+1)}_BS_{batch_size}_LR_{lr}_PS_{patch_size}_PN_{patch_num}.pth")
                
                print(f"Save the best model: {name_metric} = {'%d'%accuracy_best}")
        
        print(f"Val {name_metric}: {'%d'%accuracy_val} / {'%d'%accuracy_best}, Train {name_metric}: {'%d'%loss_train_mean}")
        
        writer.add_scalar('loss/train', loss_train_mean, epoch+1)
        writer.add_scalar('loss/val', loss_val_mean, epoch+1)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch+1)
       
        scheduler.step()
        print(f"epoch time : {'%10s'%str(datetime.datetime.now()-start_time_epoch)[:7]}")
    
    writer.close()
    print('End of training')


def gpu_ens():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


if __name__ == '__main__':
# ------------------------- spliting -------------------------
    use_gpu = 0
    name_model = 'AttU_Net'
    epochs = 500  ## 500
    batch_size = 32  ## 32
    patch_size = 128  ## 128
    patch_num = 4  ## 4
    
    lr = 0.001  ## 0.001
    step_size = 50  ## 50
    gamma = 0.5  ## 0.5
    
    epoch_show = [1, 2, 10, 50, 100]   ## [1, 2, 10, 50, 100]
    path_train = 'dataset/train'
    path_val = 'dataset/val'
# ------------------------- spliting -------------------------
    
    if len(gpu_ens()) == 1:
        device = gpu_ens()[0]
    else:
        device = gpu_ens()[use_gpu]
    print(f'The number of GPU is {len(gpu_ens())}')
    print(f'Using device {device}')

    net = AttU_Net()
    net.to(device=device)
    print(f"Network parameters: {'%.2f'%(sum(param.numel() for param in net.parameters())/1e4)} w")
    
    try:
        train_net(net = net,
                  device = device,
                  epochs = epochs,
                  name_model = name_model,
                  epoch_show = epoch_show,
                  batch_size = batch_size,
                  lr = lr,
                  step_size = step_size,
                  gamma = gamma,
                  patch_size = patch_size,
                  patch_num = patch_num,
                  path_train = path_train,
                  path_val = path_val)
    
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'bkupCheckPoint.pth')
        print('Saved interrupt checkpoint backup')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    
    print(f"the total running time: {'%10s'%str(datetime.datetime.now() - start_time)[:7]}")