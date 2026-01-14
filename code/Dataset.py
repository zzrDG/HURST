from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import torch
from torch.utils.data import Dataset

class a_Dataset(Dataset):
    def __init__(self, data,  normalize=True):
        self.normalize = normalize
        self.spatiotemporal_data = data

        self.data_min = self.data_max = None
        if self.normalize:
            self._normalize_features()
        self.feature_losses = torch.zeros(self.spatiotemporal_data.shape[-1])  
        self.feature_counts = torch.zeros(self.spatiotemporal_data.shape[-1])   

    def _normalize_features(self):

        self.spatiotemporal_data, self.data_min, self.data_max = self.data_normalization(self.spatiotemporal_data)

    def __len__(self):
        return len(self.spatiotemporal_data)

    def __getitem__(self, idx):

        return (
            self.spatiotemporal_data[idx],
            idx  # 用于验证的原始索引
        )

    @staticmethod
    def data_normalization(data):
        data_min = torch.amin(data, dim=(0, 1, 2, 3), keepdim=True)
        data_max = torch.amax(data, dim=(0, 1, 2, 3), keepdim=True)
        return (data - data_min) / (data_max - data_min + 1e-8), data_min, data_max

    def update_feature_loss(self, feature_idx, loss):
   
        self.feature_losses[feature_idx] += loss
        self.feature_counts[feature_idx] += 1

    def get_feature_selection_probs(self):

        if torch.sum(self.feature_counts) == 0:
            return torch.ones(self.spatiotemporal_data.shape[-1]) / self.spatiotemporal_data.shape[-1]  # 均匀分布

        avg_losses = self.feature_losses / (self.feature_counts + 1e-8)

       
        probs = torch.softmax(avg_losses, dim=0)  
        return probs

    def collate_fn(self, batch):

        total_features = self.spatiotemporal_data.shape[-1]
        feature_probs = self.get_feature_selection_probs().clone()  
        feature_probs[feature_probs < 0] = 0
        feature_probs[torch.isnan(feature_probs)] = 0

 
    
        feature_idx = torch.multinomial(feature_probs, 1).item()  
        
        processed = []
        for st_data, _ in batch:  
            selected_feature = st_data[..., feature_idx].unsqueeze(-1)  
            his_st = selected_feature[:7] 
            processed.append((his_st))

        
        st_batch = torch.stack(processed)    
        feature_idx_batch = torch.full((len(batch),), feature_idx)

        return st_batch, feature_idx_batch



class pretrain_Dataset(Dataset):
    def __init__(self, data, nums_of_temporal_feature=0, nums_of_spatial_feature=0, normalize=True):
        self.raw_data = data.clone()
        self.nums_temp = nums_of_temporal_feature
        self.nums_spat = nums_of_spatial_feature
        self.normalize = normalize


        self.temporal_data = self.raw_data[..., :self.nums_temp]
        self.spatial_data = self.raw_data[..., self.nums_temp:self.nums_temp+self.nums_spat]
        self.spatiotemporal_data = self.raw_data[..., self.nums_temp+self.nums_spat:]

 
        self.data_min = self.data_max = None
        if self.normalize:
            self._normalize_features()


        self.feature_losses = torch.zeros(self.spatiotemporal_data.shape[-1]) 
        self.feature_counts = torch.zeros(self.spatiotemporal_data.shape[-1])   

    def _normalize_features(self):
    
        if self.nums_temp > 0:
            self.temporal_data[..., 0] /= 31    # Day of month
            self.temporal_data[..., 1] /= 12   # Month
            self.temporal_data[..., 3] /= 6    # Day of week
            self.temporal_data[..., 4] /= 365   # Day of year
      
            self.temporal_data[..., 7:], _, _ = self.data_normalization(self.temporal_data[..., 7:])

    
        if self.nums_spat > 0:
            self.spatial_data, _, _ = self.data_normalization(self.spatial_data)

        self.spatiotemporal_data, self.data_min, self.data_max = self.data_normalization(
            self.spatiotemporal_data
        )

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):

        return (
            self.temporal_data[idx],
            self.spatial_data[idx],
            self.spatiotemporal_data[idx],
            idx  
        )

    @staticmethod
    def data_normalization(data):
        data_min = torch.amin(data, dim=(0, 1, 2, 3), keepdim=True)
        data_max = torch.amax(data, dim=(0, 1, 2, 3), keepdim=True)
        return (data - data_min) / (data_max - data_min + 1e-8), data_min, data_max

    def update_feature_loss(self, feature_idx, loss):
       
        self.feature_losses[feature_idx] += loss
        self.feature_counts[feature_idx] += 1

    def get_feature_selection_probs(self):
        
        if torch.sum(self.feature_counts) == 0:
            return torch.ones(self.spatiotemporal_data.shape[-1]) / self.spatiotemporal_data.shape[-1]  # 均匀分布


        avg_losses = self.feature_losses / (self.feature_counts + 1e-8)

     
        probs = torch.softmax(avg_losses, dim=0) 
        return probs

    def collate_fn(self, batch):

        total_features = self.spatiotemporal_data.shape[-1]
        feature_probs = self.get_feature_selection_probs()
        feature_idx = torch.multinomial(feature_probs, 1).item()  

        processed = []
        for temp, spat, st_data, _ in batch: 
    
            selected_feature = st_data[..., feature_idx].unsqueeze(-1)  # [days, x, y, 1]


            his_temp = temp[:7]    # [7, x, y, t_feat]
            his_spat = spat[:7]    # [7, x, y, s_feat]
            his_st = selected_feature[:7]  # [7, x, y, 1]

            processed.append((his_temp, his_spat, his_st))


        temp_batch = torch.stack([p[0] for p in processed])  # [B, 7, H, W, t_feat]
        spat_batch = torch.stack([p[1] for p in processed])  # [B, 7, H, W, s_feat]
        st_batch = torch.stack([p[2] for p in processed])    # [B, C, 7, H, W]

  
        feature_idx_batch = torch.full((len(batch),), feature_idx)

        return temp_batch, spat_batch, st_batch, feature_idx_batch

    def data_denormalization(self, normalized_data, feature_idx):

        if not self.normalize:
            raise ValueError("Data was not normalized")

        min_val = self.data_min[..., feature_idx].view(1, 1, 1, 1)
        max_val = self.data_max[..., feature_idx].view(1, 1, 1, 1)
        return normalized_data * (max_val - min_val) + min_val

class preDataset(Dataset):
    def __init__(self, data, normalize=True):
        self.raw_data = data.clone()  # 
        self.normalize = normalize

        self.data_min = self.data_max = None
        if self.normalize:
            self.data, self.data_min, self.data_max = self.data_normalization(self.raw_data)
        else:
            self.data = self.raw_data

        self.num_features = self.data.shape[-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       
        return self.data[idx]  

    @staticmethod
    def data_normalization(data):
        data_min = torch.amin(data, dim=(0,1,2,3))
        data_max = torch.amax(data, dim=(0,1,2,3))
        return (data-data_min)/(data_max-data_min+1e-8), data_min, data_max

    def collate_fn(self, batch):
  
        feature_idx = torch.randint(0, self.num_features, (1,)).item()
        
        processed = []
        for sample in batch:  
            selected = sample[..., feature_idx].unsqueeze(-1)  # [days, x, y, 1]
            
           
            his = selected[:7]   # [7, x, y, 1]
            pred = selected[7]    # [x, y, 1]
            
            processed.append( (his, pred) )


        his_batch = torch.stack([p[0] for p in processed])  # [B, 7, x, y, 1]
        pred_batch = torch.stack([p[1] for p in processed])  # [B, x, y, 1]
        feature_idx_batch = torch.full((len(batch),), feature_idx)  
        
        return his_batch, pred_batch, feature_idx_batch

    def data_denormalization(self, normalized_data, feature_idx):
    
        if not self.normalize:
            raise ValueError("Data was not normalized")
        
        min_val = self.data_min[feature_idx].view(1,1,1,1).to(normalized_data.device)
        max_val = self.data_max[feature_idx].view(1,1,1,1).to(normalized_data.device)
        return normalized_data * (max_val - min_val) + min_val
