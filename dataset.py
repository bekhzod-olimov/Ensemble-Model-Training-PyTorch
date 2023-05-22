# Import libraries
import torch, pandas as pd
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    
    """
    
    This class getsa data path and returns dataset to train a model.
    
    Parameter:
    
        data_path    - path to data, str.
    
    """
    
    def __init__(self, data_path):
        
        # Read csv data
        df = pd.read_csv(data_path)
        # Get images and their corresponding labels
        self.ims, self.lbls = df.iloc[:,1:].values.reshape(df.shape[0], 28, 28), df.iloc[:,0].values.reshape(df.shape[0], 1)
    
    # Get dataset length
    def __len__(self): return len(self.ims)
    
    # Get data samples based on index
    def __getitem__(self, idx): return torch.FloatTensor(self.ims[idx]).unsqueeze(0), torch.LongTensor(self.lbls[idx]).squeeze()
