# code that contains the training loop

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ode_model import *
import datetime
import os
import numpy as np
import tqdm
import json
from utils import custom_criterion, UncertaintyWeightedLoss

import torch
import torch.nn.functional as F
import pytorch_warmup as warmup

def train(model, epochs, train_loader, val_loader, model_save_path, model_name, optimizer = None, scheduler = None, chkpt_dir = None):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    
    print("Using the device: " + device)
    
    model.to(device)

    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr = 1e-3)
    else:
        optimizer = optimizer
    
    # criterion = nn.MSELoss()
    
    T_0 = int(0.02 * len(train_loader.dataset))
    T_mult = 2
    # warmup_period = int(0.005 * len(train_loader.dataset))
    warmup_period = 2000
    

    if scheduler is None:
      scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = T_0, T_mult = T_mult, eta_min = 1e-6)
      warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)
    
    history = {'Training loss': {'Total loss': [], 'Recon loss': [], 'Imputed loss': []}, # , 'KL loss': []}, 
               'Validation loss': {'Total loss': [], 'Recon loss': [], 'Imputed loss': []},
              'Last epoch': None} 

    best_vloss = 1e6
    early_stop_thresh = 15
    weighted_loss_fn = UncertaintyWeightedLoss()
    
    if chkpt_dir is not None:
      with open(os.path.join(chkpt_dir,'history.json'),'r') as f:
        history = json.load(f)
        best_vloss = np.min(history['Validation loss']['Total loss'])
        last_epoch = history['Last epoch']
        chkpt_dict = torch.load(os.path.join(chkpt_dir,model_name + '_best.pth'))
        best_epoch = chkpt_dict['best epoch']
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.mkdir(os.path.join(model_save_path, str(timestamp)))
        chkpt_dir = os.path.join(model_save_path, str(timestamp))
        last_epoch = 0
    
    for epoch in range(last_epoch,epochs):
        
        print('EPOCH {}:'.format(epoch + 1))
        
        train_dict = {'Total loss': 0,
                      'Recon loss': 0,
                      'Imputed loss': 0}
                      # 'KL loss': 0}
        
        val_dict = {'Total loss': 0,
                      'Recon loss': 0,
                      'Imputed loss': 0}
                      # 'KL loss': 0}

        model.train()
        for i, data in tqdm.tqdm(enumerate(train_loader, 0), unit="batch", total=len(train_loader)):

            spec, B1, sat, masks, spec_ori, offsets = map(lambda x: x.to(device), data)            
            optimizer.zero_grad()

            # outputs, mu, log_var = model(spec, B1, sat, masks)
            outputs = model(spec, B1, sat, masks, offsets)
            
            # loss, recon_loss, imputed_loss, kl_loss, kl_weight = custom_criterion_vae(outputs, spec_ori, masks, mu, log_var, epoch, epochs, r = 0.75, m = 20)
            loss, recon_loss, imputed_loss = custom_criterion(outputs, spec_ori, masks)
            # loss, recon_loss, imputed_loss = weighted_loss_fn(recon_loss, imputed_loss)
            
            loss.backward()
            optimizer.step()
            
            if scheduler is None:
              with warmup_scheduler.dampening():
                  if warmup_scheduler.last_step + 1 >= warmup_period:
                      scheduler.step(epoch + i / len(train_loader))
            else:
              scheduler.step(epoch + i / len(train_loader))

            train_dict['Total loss'] += loss.item()*spec.size(0)
            train_dict['Recon loss'] += recon_loss.item()*spec.size(0)
            train_dict['Imputed loss'] += imputed_loss.item()*spec.size(0)
            # train_dict['KL loss'] += kl_loss.item()*spec.size(0)

        model.eval()
        with torch.no_grad():
            for _, vdata in enumerate(val_loader):
                
                spec, B1, sat, masks, spec_ori, offsets = map(lambda x: x.to(device), vdata)
                # outputs, mu, log_var = model(spec, B1, sat, masks)
                outputs = model(spec, B1, sat, masks, offsets)
                
                # vloss, v_recon_loss, v_imputed_loss, v_kl_loss, _ = custom_criterion_vae(outputs, spec_ori, masks, mu, log_var, epoch, epochs, r = 0.75, m = 20)
                vloss, v_recon_loss, v_imputed_loss = custom_criterion(outputs, spec_ori, masks)
                # vloss, v_recon_loss, v_imputed_loss = weighted_loss_fn(v_recon_loss, v_imputed_loss)
                
                val_dict['Total loss'] += vloss.item()*spec.size(0)
                val_dict['Recon loss'] += v_recon_loss.item()*spec.size(0)
                val_dict['Imputed loss'] += v_imputed_loss.item()*spec.size(0)
                # val_dict['KL loss'] += v_kl_loss.item()*spec.size(0)
        
        history['Training loss']['Total loss'].append(train_dict['Total loss'] / len(train_loader.dataset))
        history['Training loss']['Recon loss'].append(train_dict['Recon loss'] / len(train_loader.dataset))
        history['Training loss']['Imputed loss'].append(train_dict['Imputed loss'] / len(train_loader.dataset))
        # history['Training loss']['KL loss'].append(train_dict['KL loss'] / len(train_loader.dataset))

        history['Validation loss']['Total loss'].append(val_dict['Total loss'] / len(val_loader.dataset))
        history['Validation loss']['Recon loss'].append(val_dict['Recon loss'] / len(val_loader.dataset))
        history['Training loss']['Imputed loss'].append(val_dict['Imputed loss'] / len(val_loader.dataset))
        # history['Validation loss']['KL loss'].append(val_dict['KL loss'] / len(val_loader.dataset))

        history['Last epoch'] = epoch

        val_total_loss = val_dict['Total loss'] / len(val_loader.dataset)
        
        print('Total loss train: {}, Recon loss train: {}, Imputed loss train: {}'.format(train_dict['Total loss']/len(train_loader.dataset),
                                                                                    train_dict['Recon loss']/ len(train_loader.dataset),
                                                                                    train_dict['Imputed loss']/len(train_loader.dataset)))
                                                                                    # train_dict['KL loss']/ len(train_loader.dataset),
                                                                                    # kl_weight))
               
        print('Total loss val: {}, Recon loss val: {}, Imputed loss val: {}'.format(val_dict['Total loss'] / len(val_loader.dataset),
                                                                               val_dict['Recon loss'] / len(val_loader.dataset),
                                                                               val_dict['Imputed loss']/len(val_loader.dataset)))
                                                                               # val_dict['KL loss'] / len(val_loader.dataset)))

            
        history_fpath = os.path.join(chkpt_dir,'history.json')
        with open(history_fpath,'w') as f:
            json.dump(history,f)
            
        if val_total_loss < best_vloss:
            best_vloss = val_total_loss 
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best epoch': best_epoch
                        },
                       os.path.join(chkpt_dir,model_name + '_best.pth'))

        elif epoch - best_epoch > early_stop_thresh:
            print('Early stopping stopped at epoch {}'.format(epoch))
            break
        else:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'last epoch': epoch},
                       os.path.join(chkpt_dir,model_name + '_current.pth'))
        
        scheduler.step()
        
    return model, history
        

def test(model, test_loader):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    model.to(device)

    model.eval()

    all_outputs = []
    
    with torch.no_grad():
        for data in test_loader:
            
            spec, B1, sat, masks, _, offsets = map(lambda x: x.to(device), data)
            masks = masks.bool()
            
            outputs = model(spec, B1, sat, masks, offsets)
            all_outputs.append(outputs.cpu().numpy())
            
    return np.concatenate(all_outputs, axis = 0)

def test_ffn(model, test_loader):
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    
    model.to(device)
    model.eval()
    all_outputs = []
    
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            
            all_outputs.append(outputs.cpu().numpy())
    
    return np.concatenate(all_outputs, axis = 0)