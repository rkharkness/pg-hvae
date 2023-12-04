import logging
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import auc, roc_curve
from data.luna16 import AddAnomalies

def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output, 1e-9, 1-1e-9)
    if weights is not None:
        assert len(weights) == 2
        
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

def auc_fn(y, y_pred):
    y = y.flatten()
    y_pred = y_pred.flatten()
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    return auc(fpr, tpr)
        
def grid_rec_save(img, rec, phase, root, stage, epoch):
    
    img = img.detach().cpu()
    rec = rec.detach().cpu()

    x = torch.cat((img, rec), dim=3)
    x = x / x.max()
    
    x = np.squeeze(x[0])

    plt.figure(figsize=(20,80))
    plt.gray()
    plt.subplots_adjust(0,0,1,0.95,0.01,0.01)
    for i in range(x.shape[0]):
        x_i = np.expand_dims(x[:,:,i],-1)
        plt.subplot((x.shape[0])+1//6,6,i+1), plt.imshow(x_i), plt.axis('off')
    plt.suptitle('Reconstructions', size=15)
    plt.savefig(f"{root}/{phase}_rec{epoch}_stage{stage}.png")
    plt.show()
    
    image = wandb.Image(f"{root}/{phase}_rec{epoch}_stage{stage}.png", caption='recs')
    wandb.log({f"{phase}_rec": image})
    
def grid_sample_save(x, phase, root, stage, epoch):
    x = x[0]
    x = x / x.max()
    x = np.squeeze(x)

    plt.figure(figsize=(20,35))
    plt.gray()
    plt.subplots_adjust(0,0,1,0.95,0.01,0.01)
    for i in range(x.shape[0]):
        x_i = np.expand_dims(x[:,:,i],-1)
        plt.subplot((x.shape[0])+1//10,10,i+1), plt.imshow(x_i), plt.axis('off')
    plt.suptitle('Generated images', size=15)
    plt.savefig(f"{root}/{phase}_gen{epoch}_{stage}.png")
    plt.show()
    
    image = wandb.Image(f"{root}/{phase}_gen{epoch}_{stage}.png", caption='noise gen.')
    wandb.log({f"{phase}_gen": image})

def train(model, dataloader, bs, optimizer, pg_stage, opt):

    keep_training = True
    logger = model.logger

    log_path = os.path.join(opt.model_dir,'log.csv')
    save_path = os.path.join(opt.model_dir, 'models', f'./stage_{pg_stage}.pth') #TODO add stage to name
    
    img_save_path = os.path.join(model.root,'imgs')
    os.makedirs(img_save_path, exist_ok=True)  # Create the directory if it doesn't exist

    epoch = 0 # TODO get from checkpoint
    patience = 5
    accumulation_steps = 24 // bs
    
    add_ano = AddAnomalies(max_anomaly_size = 8)
    bce_loss = nn.BCELoss()
    # scaler = GradScaler()

    while keep_training:
        epoch += 1
        
        kl_anneal = min(5, opt.kl_weight*(epoch/10))
        bce_anneal = min(opt.bce_weight, (opt.bce_weight*epoch/1000))
        wandb.log({"kl-weight":kl_anneal})
        
        # classify = False
        
        # if pg_stage > 1:
            # if epoch > 10:
        classify = True
        
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.epochs - epoch, eta_min=1e-4)

        logger.info('-' * 10)
        logger.info('Epoch {}/'.format(epoch))
        
        best_val_loss = np.inf

        for phase in ['train', 'val']:
            logger.info(phase)
            if phase == 'train':
                model.train()
            
            running_loss = 0.0
            running_loss_img = 0.0
            running_kl = 0.0
            running_kl_multi = []
            running_class = 0.0
            running_loss_adv = 0.0
            epoch_samples = 0
            
            running_auc = 0.0
            
            y_true_list = []
            y_pred_list = []
            
            with torch.set_grad_enabled(phase == 'train'):
                

                for idx, batch in tqdm(enumerate(dataloader[phase])):
                    epoch_samples += 1

                    img = batch['ct']['data']
                    y = batch['ct_class']
                    y = torch.unsqueeze(y, 0)
                    
                    if y == 1:
                        img = add_ano(img)
                    
                    img = img.to(device=opt.device)
                    y = y.to(dtype=torch.float32, device=opt.device)
                    
                    output_img, kls, z = model(img, pg_stage)
                    pred_y = model.classifier(z)
                                        
                    y_pred_list.extend(pred_y.detach().cpu().numpy().astype(float))
                    y_true_list.extend(y.detach().cpu().numpy().astype(int))                 
                    
                    class_loss = bce_loss(pred_y, y)

                    kl = torch.sum(torch.stack(kls['prior']))
                    loss_img = nn.SmoothL1Loss(reduction='sum')(output_img, img)
                    loss = loss_img + (opt.kl_weight*(opt.beta * kl))
                    
                    if classify:
                        loss += (class_loss*opt.bce_weight)
                            
                    running_loss += loss.item()
                    running_loss_img += loss_img.item()
                    running_kl += kl.item()
                    running_class += class_loss.item()

                    
                    if phase == 'train':                   
                        loss = loss / accumulation_steps
                        loss.backward()
                        
                        if (idx + 1) % accumulation_steps == 0:
                            # Perform a weight update every `accumulation_steps` mini-batches
                            optimizer.step()
                            optimizer.zero_grad()

                    if idx == 1:
                        grid_rec_save(img, output_img, phase, img_save_path, pg_stage, epoch)

            auc_score = auc_fn(np.array(y_true_list), np.array(y_pred_list))

            logger.info('{}  Loss Total: {:.4f}'.format(
                phase, running_loss / epoch_samples))

            logger.info('{}  Loss Img: {:.4f}'.format(
                phase, running_loss_img / epoch_samples))

            logger.info('{}  Loss KL: {:.4f}'.format(
                phase, running_kl / epoch_samples))
            
            logger.info('{}  Loss Classifier: {:.4f}'.format(
                phase, running_class / epoch_samples))
            
            logger.info('{}  AUC: {:.4f}'.format(
                phase, auc_score))            
            
            wandb.log(
            {
                f"{phase}/total_loss" : running_loss/epoch_samples,
                f"{phase}/kl_loss" :running_kl/epoch_samples,
                f"{phase}/rec_loss" :running_loss_img/epoch_samples,
                f"{phase}/class_loss": running_class/epoch_samples,
                f"{phase}/auc":auc_score
                }
            )

            epoch_loss = running_loss / epoch_samples
               
            if phase == 'val':
                gen = model.sample(img.shape[0])
                gen = gen.detach().cpu()
                grid_sample_save(gen, phase, img_save_path, pg_stage, epoch)
                
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    patience = 15
                    model.model_save(pg_stage)
                    wandb.save("model_weights.pth")

                else:
                    patience -= 1
                    if patience == 0:
                        keep_training = False
                        break
                
            if epoch==800:
                keep_training=False
 
                    
                    