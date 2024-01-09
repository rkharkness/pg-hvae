from sklearn.model_selection import train_test_split
import pandas as pd
import torchio as tio
import numpy as np
import torch
import glob 
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, WeightedRandomSampler

# torch.set_default_dtype(torch.float16)

class AddAnomalies(object):
    def __init__(self, 
                 max_anomalies = 1, 
                 max_anomaly_size = 10,
                ):
        
        super().__init__()
        
        self.max_anomalies = max_anomalies
        self.max_anomaly_size = max_anomaly_size
        

    def __call__(self, image):
        """
        Add anomalies to a medical image by overlaying random shapes/masks.

        Parameters:
        image (torch.Tensor): Original medical image.
        num_anomalies (int): Number of anomalies to add.
        max_anomaly_size (int): Maximum size of each anomaly.

        Returns:
        torch.Tensor: Image with added anomalies.
        """
        # Copy the original image to avoid modifying it
        
        image_with_anomalies = image.clone()

        _, _, height, width, depth = image.shape
        num_anomalies = random.randint(1, self.max_anomalies)

        for _ in range(num_anomalies):
            anomaly_size = random.randint(3, self.max_anomaly_size)

            # Randomly choose a pixel value for the anomaly
            anomaly_value = random.randint(80, 180)
            anomaly_value /= 256

            # Randomly choose the position for the anomaly
            position_x = random.randint(0, width - anomaly_size)
            position_y = random.randint(0, height - anomaly_size)
            position_z = random.randint(0, depth - anomaly_size)


            # Create a random mask/anomaly
            anomaly = torch.full((anomaly_size, anomaly_size, anomaly_size), anomaly_value, dtype=torch.float)

            # Overlay the anomaly on the image
            image_with_anomalies[:,:, position_y:position_y+anomaly_size,
                                 position_x:position_x+anomaly_size, position_z:position_z+anomaly_size] = anomaly

        return image_with_anomalies
    
    
def create_sampler(data, class_weights=[1/20,1/50]):
    sample_weights = [0] * len(data)

    for idx, (data, label) in enumerate(zip(data['path'].values, data['class'].values)):
        sample_weights[idx] = class_weights[label]    
        
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
                                            
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

def get_dataloader(x, id_, class_, train, batch_size, sampler, img_size=(64,64,64), num_workers=64):
    dataset = trainset_io(x, id_, class_, img_size, train=train)
   
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,\
                                             pin_memory=True, drop_last=True, sampler=sampler)
    return dataloader

def collect_paths(root):
    paths = [glob.glob(root+f"/subset{i}/*mhd") for i in range (10)]
    return paths
    
def trainset_io(X, ID, C, img_size, train=True):
    """
    To create a 3D train dataset with TorchIO library
    Intensity rescaling between [0,1] for the CT air and bone HU values
    """
    # sample_dict = {32 * i: tuple(x / (2 ** i) for x in (10.8, 12, 10.8)) for i in range(30)}
    sample_dict = {64:(5.4,6.,5.4), 96:(4.,4.5,4.), 128: (2.7,3,2.7), 192:(2.4,2.8,2.4), 256:(1.3,1.5,1.3), 512:(0.65,0.75,0.65), 1024:(0.65,0.75,0.65)}
    resample_vals = sample_dict[img_size[0]]
    HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1024, 800
    transform = tio.Compose([
          tio.transforms.Clamp(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE),
          tio.transforms.ToCanonical(),
          tio.CopyAffine('ct'),
          tio.transforms.Resample(resample_vals), # spacing of scans after spacing unification
          tio.transforms.CropOrPad(img_size), #(480, 480, 280)),
          # tio.Mask(masking_method='lungs'),
          tio.transforms.RescaleIntensity(out_min_max = [0,1]), 
    ]) 

    subjects_list = []
    for (ct, id_, class_) in zip(X, ID, C):
        subject = tio.Subject(
            ct=tio.ScalarImage(path = ct,type="intensity"),
            seriesuid = id_,
            ct_class = class_
            )
        
        subject = transform(subject)
            
        subjects_list.append(subject)
            
    return tio.SubjectsDataset(subjects_list) #, load_getitem = True)


if __name__ == "__main__":
    
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
    train_data = train_data.sample(frac=1)[:50]
    val_data = data[data['split']=='val'].reset_index(drop=True)
    val_data = val_data.sample(frac=1)[:2]

    x_train = train_data['path'].values
    id_train = train_data['seriesuid'].values
    class_train = train_data['class'].values
    
    print(train_data['class'].value_counts())
    
    train_sampler = create_sampler(train_data, class_weights=[2.5,1])                                    

    x_val = val_data['path'].values
    id_val = val_data['seriesuid'].values
    class_val = val_data['class'].values
    
    print(val_data['class'].value_counts())

                                        
    val_sampler = create_sampler(val_data, class_weights=[2.5,1])

    add_ano = AddAnomalies(max_anomaly_size = 12)
                                        

    BATCHSIZE = 1
    IMG_SIZE = 64
    train_loader = get_dataloader(x_train, id_train, class_train, train=True, batch_size=BATCHSIZE, \
                                  img_size=(IMG_SIZE,IMG_SIZE,IMG_SIZE), sampler=train_sampler, num_workers=4)
    
    val_loader = get_dataloader(x_val, id_val, class_val, train=True, batch_size=BATCHSIZE,\
                                  img_size=(IMG_SIZE,IMG_SIZE,IMG_SIZE), sampler=val_sampler, num_workers=4)

    image_data = next(iter(val_loader))
    
    y = image_data['ct_class']           
    x = image_data['ct']['data']
    
    if y == 1:
        x = add_ano(x)

    x1 = x
    
    x = torch.cat((x, x1), dim=3)
    x = x / x.max()
    x = np.squeeze(x)
    
    plt.figure(figsize=(42,48))
    plt.gray()
    plt.subplots_adjust(0,0,1,0.95,0.01,0.01)
    for i in range(x.shape[0]):
        x_i = np.expand_dims(x[:,:,i],-1)
        plt.subplot((x.shape[0]//10)+1,10,i+1), plt.imshow(x_i), plt.axis('off')
    plt.suptitle(f'Lung CT-scan mha (raw) files - {y}', size=15)
    plt.savefig("./batch.png")
    plt.show()
    
    for data in train_loader:
        print(data['ct_class'])
    