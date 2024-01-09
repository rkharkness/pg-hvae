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
from torchinfo import summary
# torch.set_default_dtype,(torch.float16)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Main script args.")

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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_num_features", type=int, default=4)
    parser.add_argument("--max_features", type=int, default=12)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--resume_stage", type=int, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)

 
    # Parse the command-line arguments
    args = parser.parse_args()
    
    if args.resume:
        assert args.resume_stage is not None, "Must define start stage if resuming training"
        assert args.resume_dir is not None, "Must define path for model weight loading if resuming training"

    
    wandb.init(project="pg-hvae", entity="rachaelharkness1")
    
    wandb.init(config={
        "kl_weight": args.kl_weight, 
        "beta": args.beta, 
        "batch_size":args.batch_size, 
        "lr":args.lr,
        "bce_weight":args.bce_weight,
        "base_num_features":args.base_num_features,
        "max_features":args.max_features,
        "dirichlet":args.dirichlet,
        "pc_grad":args.pc_grad
                      })
    
    pathlist = collect_paths("/root/data/luna16")
    pathlist = [i for paths in pathlist for i in paths]
    seriesuid = [i.split("/")[-1][:-4] for i in pathlist]
    
    candidates = pd.read_csv("/root/data/luna16/candidates.csv")
    candidates = candidates.groupby("seriesuid").sum('class')
    candidates['class'] = [1 if i > 0 else 0 for i in candidates['class']]
    print(candidates['class'].value_counts())
    
    data = pd.DataFrame()
    data['path'] = pathlist
    data['seriesuid'] = seriesuid
    
    data = data.merge(candidates, on="seriesuid")
    
    data = split_data(data)

    train_data = data[data['split']=='train'].reset_index(drop=True)
    print(train_data['class'].value_counts())
    val_data = data[data['split']=='val'].reset_index(drop=True)
    print(val_data['class'].value_counts())

    x_train = train_data['path'].values
    id_train = train_data['seriesuid'].values
    class_train = train_data['class'].values

    x_val = val_data['path'].values
    id_val = val_data['seriesuid'].values
    class_val = val_data['class'].values
    
    # logger = create_logger("/root/code/pg_hvae/")

    res = 64
    model = BIVA3D(stochastic_module=None, dirichlet=args.dirichlet, dir_conc=args.dir_conc, resume=args.resume, root=args.resume_dir,input_shape=(res,res,res), base_num_features=args.base_num_features, max_features=args.max_features, num_pool=4)
    model.cuda()
    
    if args.resume_stage != None:
        if args.resume_stage > 1:
            assert args.resume, "Must indicate resuming if trying to start with a later stages"
    
    if args.resume:
        start_stage = args.resume_stage
        assert args.resume_stage != None, "Must define training stage if resuming"
    else:
        start_stage = 1
    
    max_save_stage = 0

    if args.resume_dir != None:
        files = os.listdir(args.resume_dir + "/models")
        if len(files) > 0:
            saved_stages = list(filter(str.isdigit, str(files)))
            max_save_stage = max([int(i) for i in saved_stages])

        
    for pg_stage in range(start_stage,6):
        
        if start_stage > 1: # Resuming
            extra_stages = start_stage - 1  
            while extra_stages >= 1: # Grow to the previous stage for resume
                load_stage = (start_stage - extra_stages) + 1 # 1
                model.grow(load_stage)
                res = res*2
                extra_stages-=1
            
            if max_save_stage == args.resume_stage:
                # Load weight for if resuming same resolution stage
                print("loading weight for same stage")
                model = model.model_load(start_stage)
            
            model.cuda()
            
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adamax(params, lr=args.lr, betas=(0.9, 0.999,))
                
        print(f"stage {pg_stage} | image resolution - ({res},{res},{res})")
        
        all_ = [p for p in model.parameters()]
        print(f"training {len(params)} / {len(all_)} params")
        
        train_sampler = create_sampler(train_data, class_weights=[2.2,1])                                    
        val_sampler = create_sampler(val_data, class_weights=[3,1])
        
        bs = 1
        train_loader = get_dataloader(x_train, id_train, class_train, train=True, batch_size=bs, img_size=(res,res,res), sampler=train_sampler, num_workers=12)
        val_loader = get_dataloader(x_val, id_val, class_val, train=True, batch_size=bs, img_size=(res,res,res), sampler=val_sampler, num_workers=12)

        dataloader = {"train":train_loader, "val":val_loader}
        # model.load_state_dict(torch.load("/root/code/pg_hvae/runs/2023-11-20_12-45-37/models/model1-classifier.pth"))

        train(model, dataloader, bs, optimizer, pg_stage, opt=args)

        model.grow(pg_stage)

        model.cuda()
        
        res = res*2 # Double img resolution at end of stage
        
        
        
        
        

        
