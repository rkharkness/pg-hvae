from model.biva import BIVA3D
from model.classifier import MLP

from data.luna16 import split_data, get_dataloader, collect_paths, create_sampler
from sklearn.model_selection import train_test_split
import pandas as pd
from training import train
import logging
import torch
import argparse
import os
import wandb

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Hyper-param. tuning script args.")

    # Define command-line arguments
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--model_dir", type=str, default="/root/code/pg_hvae", help="Path to the output file")
    parser.add_argument("--dirichlet", type=bool, default=True)
    parser.add_argument("--dir_conc", type=float, default=0.8)
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--kl_weight", type=int, default=1e-9)
    parser.add_argument("--bce_weight", type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--pc_grad", type=bool, default=False)
    parser.add_argument("--base_num_features", type=int, default=4)
    parser.add_argument("--max_features", type=int, default=12)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--resume_stage", type=int, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)
    
# lr
 
    # Parse the command-line arguments
    args = parser.parse_args()
    
    assert args.resume == False, "Must not resume in hyperparameter tuning"


    pathlist = collect_paths("/root/data/luna16")
    pathlist = [i for paths in pathlist for i in paths]
    seriesuid = [i.split("/")[-1][:-4] for i in pathlist]

    candidates = pd.read_csv("/root/data/luna16/candidates.csv")
    candidates = candidates.groupby("seriesuid").sum('class')
    candidates['class'] = [1 if i > 0 else 0 for i in candidates['class']]

    data = pd.DataFrame()
    data['path'] = pathlist
    data['seriesuid'] = seriesuid

    data = data.merge(candidates, on="seriesuid")

    data = split_data(data)

    train_data = data[data['split']=='train'].reset_index(drop=True)
    val_data = data[data['split']=='val'].reset_index(drop=True)

    x_train = train_data['path'].values
    id_train = train_data['seriesuid'].values
    class_train = train_data['class'].values

    x_val = val_data['path'].values
    id_val = val_data['seriesuid'].values
    class_val = val_data['class'].values

    train_sampler = create_sampler(train_data, class_weights=[2.2,1])                                    
    val_sampler = create_sampler(val_data, class_weights=[3,1])

    bs = 1
    res = 64

    train_loader = get_dataloader(x_train, id_train, class_train, train=True, batch_size=bs, img_size=(res,res,res), sampler=train_sampler, num_workers=12)
    val_loader = get_dataloader(x_val, id_val, class_val, train=True, batch_size=bs, img_size=(res,res,res), sampler=val_sampler, num_workers=12)
    dataloader = {"train":train_loader, "val":val_loader}

    def objective(dataloader, bs, optimizer, pg_stage, config, opt):
        model = BIVA3D(stochastic_module=None, dirichlet=config.dirichlet, dir_conc=config.dir_conc, resume=config.resume,\
                       root=config.resume_dir,input_shape=(res,res,res), base_num_features=config.base_num_features,\
                       max_features=config.max_features, num_pool=4)
        
        model.cuda()

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adamax(params, lr=args.lr, betas=(0.9, 0.999,))
        train(wandb.config, model, dataloader, bs, optimizer, pg_stage, opt)
        
#TODO sort out conflict between wandb.connfig and opt


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val/class_loss"},
    "parameters": { #

        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}