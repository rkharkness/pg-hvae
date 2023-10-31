import logging
import torch
import os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from torch.cuda.amp import GradScaler, autocast


        
def grid_rec_save(img, rec, phase, stage, epoch):
    
    img = img.detach().cpu()
    rec = rec.detach().cpu()

    x = torch.cat((img, rec), dim=3)
    x = x / x.max()
    
    x = np.squeeze(x[0])

    plt.figure(figsize=(42,48))
    plt.gray()
    plt.subplots_adjust(0,0,1,0.95,0.01,0.01)
    for i in range(x.shape[0]):
        x_i = np.expand_dims(x[:,:,i],-1)
        plt.subplot((x.shape[0])+1//10,10,i+1), plt.imshow(x_i), plt.axis('off')
    plt.suptitle('Reconstructions', size=15)
    plt.savefig(f"./{phase}_rec{epoch}_stage{stage}.png")
    plt.show()
    
    image = wandb.Image(f"./{phase}_rec{epoch}.png", caption='recs')
    wandb.log({f"{phase}_rec": image})
    
def grid_sample_save(x, phase, stage, epoch):
    x = x[0]
    x = x / x.max()
    x = np.squeeze(x)

    plt.figure(figsize=(42,48))
    plt.gray()
    plt.subplots_adjust(0,0,1,0.95,0.01,0.01)
    for i in range(x.shape[0]):
        x_i = np.expand_dims(x[:,:,i],-1)
        plt.subplot((x.shape[0])+1//10,10,i+1), plt.imshow(x_i), plt.axis('off')
    plt.suptitle('Generated images', size=15)
    plt.savefig(f"./{phase}_gen{epoch}_{stage}.png")
    plt.show()
    
    image = wandb.Image(f"./{phase}_gen{epoch}.png", caption='noise gen.')
    wandb.log({f"{phase}_gen": image})

def train(model, dataloader, bs, logger, optimizer, pg_stage, opt):

    keep_training = True

    log_path = os.path.join(opt.model_dir,'log.csv')
    save_path = os.path.join(opt.model_dir, 'models', f'./stage_{pg_stage}.pth') #TODO add stage to name

    epoch = 0 # TODO get from checkpoint
    patience = 5
    accumulation_steps = 24 // bs
    scaler = GradScaler()

    kl_anneal = min(1e-3, opt.kl_weight*(epoch*10))
    wandb.log({"kl-weight":kl_anneal})

    while keep_training:
        epoch += 1
        
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
            running_loss_adv = 0.0
            epoch_samples = 0
            
            with torch.set_grad_enabled(phase == 'train'):
                
                for idx, batch in tqdm(enumerate(dataloader[phase])):
                    epoch_samples += 1

                    img = batch['ct']['data']
                    img = img.to(dtype=torch.float16, device=opt.device)
                    
                    with autocast():
                        output_img, kls  = model(img, pg_stage)

                        kl = torch.sum(torch.stack(kls['prior']))
                        loss_img = nn.L1Loss(reduction='sum')(output_img, img)
                        loss = loss_img + (opt.kl_weight*(opt.beta * kl))
                    
                    running_loss += loss.item()
                    running_loss_img += loss_img.item()
                    running_kl += kl.item()

                    
                    if phase == 'train':                   
                        loss = loss / accumulation_steps
                        scaler.scale(loss).backward()
                        
                        if (idx + 1) % accumulation_steps == 0:
                            # Perform a weight update every `accumulation_steps` mini-batches
                            scaler.step(optimizer)
                            scaler.update()
                    
                    
                    if idx == 1:
                        grid_rec_save(img, output_img, phase, pg_stage, epoch)

            logger.info('{}  Loss Total: {:.4f}'.format(
                phase, running_loss / epoch_samples))

            logger.info('{}  Loss Img: {:.4f}'.format(
                phase, running_loss_img / epoch_samples))

            logger.info('{}  Loss KL: {:.4f}'.format(
                phase, running_kl / epoch_samples))
            
            wandb.log(
            {
                f"{phase}/total_loss" : running_loss/epoch_samples,
                f"{phase}/kl_loss" :running_kl/epoch_samples,
                f"{phase}/rec_loss" :running_loss_img/epoch_samples
                }
            )

            epoch_loss = running_loss / epoch_samples
               
            if phase == 'val':
                gen = model.sample(img.shape[0])
                gen = gen.detach().cpu()
                grid_sample_save(gen, phase, pg_stage, epoch)
                
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    patience = 15
                    torch.save(model.state_dict(), save_path)
                else:
                    patience -= 1
                    if patience == 0:
                        keep_training = False
                        break
            
            # if epoch == 200 and pg_stage==6:
            #     keep_training=False
                
            if epoch==20:
                keep_training=False
 
                    
                    