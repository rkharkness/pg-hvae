from sklearn.model_selection import train_test_split
import pandas as pd
import torchio as tio
import numpy as np
import torch
import glob 
import matplotlib.pyplot as plt

def split_data(df):
    
    # Define the groups based on a column in the DataFrame
    groups = df['seriesuid'].unique()

    # Split the groups into train and remaining groups
    train_groups, remaining_groups = train_test_split(groups, test_size=0.2, random_state=42)

    # Split the remaining groups into validation and test groups
    val_groups, test_groups = train_test_split(remaining_groups, test_size=0.5, random_state=42)

    # Create train, validation, and test dataframes based on the selected groups
    train_df = df[df['seriesuid'].isin(train_groups)]
    val_df = df[df['seriesuid'].isin(val_groups)]
    test_df = df[df['seriesuid'].isin(test_groups)]

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    df = pd.concat([train_df, val_df, test_df])
    
    # Check no overlap
    train_ids = df[df['split']=='train']['seriesuid'].values
    val_ids = df[df['split']=='val']['seriesuid'].values
    test_ids = df[df['split']=='test']['seriesuid'].values

    if train_ids in val_ids or train_ids in test_ids or val_ids in test_ids:
        print('ERROR - OVERLAP IN SPLITTING DATA')
    
    return df

def get_dataloader(x, id_, train, batch_size, sample_weight, num_workers=64):
    dataset = trainset_io(x, id_, train=train)
   
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)
    return dataloader

def collect_paths(root):
    paths = [glob.glob(root+f"/subset{i}/*mhd") for i in range (10)]
    return paths
    
def trainset_io(X, ID, train=True):
    """
    To create a 3D train dataset with TorchIO library
    Intensity rescaling between [0,1] for the CT air and bone HU values
    """
    HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, -700
    transform = tio.Compose([
          tio.transforms.Clamp(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE),
          tio.transforms.ToCanonical(),
          tio.CopyAffine('ct'),
          tio.transforms.Resample((5.5,6.,5.5)), # spacing of scans after spacing unification
          tio.transforms.CropOrPad((64, 64, 64)), #(480, 480, 280)),
          # tio.Mask(masking_method='lungs'),
          tio.transforms.RescaleIntensity(out_min_max = [0,1]), 
    ]) 

    subjects_list = []
    for (ct, id_) in zip(X, ID):
        subject = tio.Subject(
            ct=tio.ScalarImage(path = ct,type="intensity"),
            seriesuid = id_
            )
        
        subject = transform(subject)
            
        subjects_list.append(subject)
            
    return tio.SubjectsDataset(subjects_list, load_getitem = True)


if __name__ == "__main__":
    
    pathlist = collect_paths("/root/data/luna16")
    pathlist = [i for paths in pathlist for i in paths]
    seriesuid = [i.split("/")[-1][:-4] for i in pathlist]
    
    data = pd.DataFrame()
    data['path'] = pathlist
    data['seriesuid'] = seriesuid
    
    data = split_data(data)

    train_data = data[data['split']=='train'].reset_index(drop=True)
    val_data = data[data['split']=='val'].reset_index(drop=True)

    x_train = train_data['path'].values[:3]
    id_train = train_data['seriesuid'].values[:3]

    x_val = val_data['path'].values[:3]
    id_val = val_data['seriesuid'].values[:3]

    
    BATCHSIZE=1
    train_loader = get_dataloader(x_train, id_train, train=True, batch_size=BATCHSIZE, sample_weight=None, num_workers=4)
    val_loader = get_dataloader(x_val, id_val, train=True, batch_size=BATCHSIZE, sample_weight=None, num_workers=4)

    image_data = next(iter(val_loader))
    
    x = image_data['ct']['data']
    x1 = x
    
    x = torch.cat((x, x1), dim=3)

    x = x / x.max()
    
    x = np.squeeze(x)
    

    plt.figure(figsize=(28,32))
    plt.gray()
    plt.subplots_adjust(0,0,1,0.95,0.01,0.01)
    for i in range(x.shape[0]):
        x_i = np.expand_dims(x[:,:,i],-1)
        plt.subplot(16,10,i+1), plt.imshow(x_i), plt.axis('off')
    plt.suptitle('Lung CT-scan mha (raw) files', size=15)
    plt.savefig("./batch.png")
    plt.show()
    