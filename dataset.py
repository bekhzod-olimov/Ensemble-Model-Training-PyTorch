import torch, pandas as pd
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    
    def __init__(self, data_path):
        
        df = pd.read_csv(data_path)
        self.ims = df.iloc[:,1:].values.reshape(df.shape[0], 28, 28)
        self.lbls = df.iloc[:,0].values.reshape(df.shape[0], 1)
    
    def __len__(self): return len(self.ims)
    
    def __getitem__(self, idx): return torch.FloatTensor(self.ims[idx]).unsqueeze(0), torch.LongTensor(self.lbls[idx]).squeeze()
