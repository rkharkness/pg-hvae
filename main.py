from model.biva import BIVA3D
from data.luna16 import split_data, get_dataloader, collect_paths

from sklearn.model_selection import train_test_split
import pandas as pd
from training import train
import logging
import torch
import argparse
import os
import wandb
from torchinfo import summary
torch.set_default_dtype,(torch.float16)

class WandbHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        wandb.run.log({record.levelname: msg})
        
def create_logger(folder):
    """Create a logger to save logs."""
    compt = 0
    while os.path.exists(os.path.join(folder,f"logs_{compt}.txt")):
        compt+=1
    logname = os.path.join(folder,f"logs_{compt}.txt")
    
    logger = logging.getLogger()
    fileHandler = logging.FileHandler(logname, mode="w")
    consoleHandler = logging.StreamHandler()
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    formatter = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(WandbHandler())

    return logger 



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Main script args.")

    # Define command-line arguments
    parser.add_argument("--epochs", type=int, default=20 )
    parser.add_argument("--model_dir", type=str, default="/root/code/pg_hvae", help="Path to the output file")
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--kl_weight", type=int, default=1e-4)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--base_num_features", type=int, default=128)


    # Parse the command-line arguments
    args = parser.parse_args()
    
    wandb.init(project="pg-hvae", entity="rachaelharkness1")
    
    wandb.init(config={
        "kl_weight": args.kl_weight, "beta": args.beta, "batch_size":args.batch_size, "lr":args.lr
                      })
    # wandb.config.update({"backbone_type": "resnet", "channels": 16})
    
    pathlist = collect_paths("/root/data/luna16")
    pathlist = [i for paths in pathlist for i in paths]
    seriesuid = [i.split("/")[-1][:-4] for i in pathlist]
    
    data = pd.DataFrame()
    data['path'] = pathlist
    data['seriesuid'] = seriesuid
    
    data = split_data(data)

    train_data = data[data['split']=='train'].reset_index(drop=True)    
    val_data = data[data['split']=='val'].reset_index(drop=True)

    x_train = train_data['path'].values
    id_train = train_data['seriesuid'].values

    x_val = val_data['path'].values
    id_val = val_data['seriesuid'].values
    
    res = 64
    model = BIVA3D(stochastic_module=None, input_shape=(res,res,res), base_num_features=16, num_pool=4)
    model.cuda()
    
    logger = create_logger("/root/code/pg_hvae/")
        
    for pg_stage in range(1,6):        
        
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adamax(params, lr=args.lr, betas=(0.9, 0.999,))

        print(f"stage {pg_stage} | image resolution - ({res},{res},{res})")
        
        all_ = [p for p in model.parameters()]
        print(f"training {len(params)} / {len(all_)} params")
        
        bs = 1
        
        train_loader = get_dataloader(x_train, id_train, train=True, batch_size=bs, img_size=(res,res,res),
                                      sample_weight=None, num_workers=48)
        val_loader = get_dataloader(x_val, id_val, train=True, batch_size=bs, img_size=(res,res,res), sample_weight=None,
                                    num_workers=48)
        

        dataloader = {"train":train_loader, "val":val_loader}
        train(model, dataloader, bs, logger, optimizer, pg_stage, opt=args)
                
        previous_path = f"/root/code/pg_hvae/models/stage_{pg_stage}.pth"
        prev_model = torch.load(previous_path)

        model.grow(pg_stage)
        model.cuda()
        
        res = res*2
        
        
        
        
        

        
