import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import scipy.io as sio

class BrainGraphDataset(Dataset):
    def __init__(self, data_path, num_nodes_wm=48, num_nodes_gm=18, k_fold=10):
        self.data_path = data_path
        self.is_hcp = 'HCP' in data_path
        
        if self.is_hcp:
            self.num_nodes_wm = 48
            self.num_nodes_gm = 82
            print("Loading HCP dataset with GM nodes: 82, WM nodes: 48")
        else:
            self.num_nodes_wm = num_nodes_wm
            self.num_nodes_gm = num_nodes_gm
            print(f"Loading regular dataset with GM nodes: {num_nodes_gm}, WM nodes: {num_nodes_wm}")
            
        self.k_fold = k_fold
        self.data = self._load_data()
        print(f"Total valid samples after filtering NaN: {len(self.data)}")
        self.k_fold_split = self._get_kfold_splits()

    def compute_correlation_matrix(self, time_series):
        correlation_matrix = np.corrcoef(time_series)
        np.fill_diagonal(correlation_matrix, 0)
        return correlation_matrix

    def _check_valid_data(self, wm_time, gm_time, connectivity):
        if (np.isnan(wm_time).any() or 
            np.isnan(gm_time).any() or 
            np.isnan(connectivity).any() or 
            np.isinf(wm_time).any() or 
            np.isinf(gm_time).any() or 
            np.isinf(connectivity).any()):
            return False
        return True

    def _load_data(self):
        data_list = []
        
        if self.is_hcp:
            male_path = os.path.join(self.data_path, 'MALE')
            male_files = [f for f in os.listdir(male_path) if f.endswith('.mat')]
            for file in male_files:
                mat_data = sio.loadmat(os.path.join(male_path, file))
                wm_time = mat_data['Eve_time']  # 48*1190
                gm_time = mat_data['Brd_time']  # 82*1190
                connectivity = mat_data['matr']  # 48*82
                
                if self._check_valid_data(wm_time, gm_time, connectivity):
                    wm_corr = self.compute_correlation_matrix(wm_time) 
                    gm_corr = self.compute_correlation_matrix(gm_time)  
                    data_list.append({
                        'wm_time': wm_corr, 
                        'gm_time': gm_corr,
                        'connectivity': connectivity,
                        'label': 0  # MALE
                    })
            
            female_path = os.path.join(self.data_path, 'FEMALE')
            female_files = [f for f in os.listdir(female_path) if f.endswith('.mat')]
            for file in female_files:
                mat_data = sio.loadmat(os.path.join(female_path, file))
                wm_time = mat_data['Eve_time']
                gm_time = mat_data['Brd_time']
                connectivity = mat_data['matr']
                
                if self._check_valid_data(wm_time, gm_time, connectivity):
                    wm_corr = self.compute_correlation_matrix(wm_time)
                    gm_corr = self.compute_correlation_matrix(gm_time)
                    
                    data_list.append({
                        'wm_time': wm_corr,
                        'gm_time': gm_corr,
                        'connectivity': connectivity,
                        'label': 1  # FEMALE
                    })
        
        else:
            hc_path = os.path.join(self.data_path, 'HC')
            hc_files = [f for f in os.listdir(hc_path) if f.endswith('.mat')]
            for file in hc_files:
                mat_data = sio.loadmat(os.path.join(hc_path, file))
                wm_time = mat_data['Eve_time']  
                gm_time = mat_data['Brd_time']  
                connectivity = mat_data['matr'] 
                
                if self._check_valid_data(wm_time, gm_time, connectivity):
                    wm_corr = self.compute_correlation_matrix(wm_time)  
                    gm_corr = self.compute_correlation_matrix(gm_time)  
                    data_list.append({
                        'wm_time': wm_corr,
                        'gm_time': gm_corr,
                        'connectivity': connectivity,
                        'label': 0
                    })
            
            mdd_path = os.path.join(self.data_path, 'MDD')
            mdd_files = [f for f in os.listdir(mdd_path) if f.endswith('.mat')]
            for file in mdd_files:
                mat_data = sio.loadmat(os.path.join(mdd_path, file))
                wm_time = mat_data['Eve_time']
                gm_time = mat_data['Brd_time']
                connectivity = mat_data['matr'] 
                
                if self._check_valid_data(wm_time, gm_time, connectivity):
                    wm_corr = self.compute_correlation_matrix(wm_time)  
                    gm_corr = self.compute_correlation_matrix(gm_time)  
                    data_list.append({
                        'wm_time': wm_corr,
                        'gm_time': gm_corr,
                        'connectivity': connectivity,
                        'label': 1
                    })
        
        return data_list
    
    def _get_kfold_splits(self):
        kf = KFold(n_splits=self.k_fold, shuffle=True, random_state=42)
        return list(kf.split(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'wm_time': torch.FloatTensor(item['wm_time']),
            'gm_time': torch.FloatTensor(item['gm_time']),
            'connectivity': torch.FloatTensor(item['connectivity']),
            'label': torch.LongTensor([item['label']])
        }

    def kfold_split(self, batch_size, test_index):
        assert test_index < self.k_fold
        valid_index = test_index
        test_split = self.k_fold_split[test_index][1]
        valid_split = self.k_fold_split[valid_index][1]

        train_mask = np.ones(len(self.data))
        train_mask[test_split] = 0
        train_mask[valid_split] = 0
        train_split = train_mask.nonzero()[0]

        train_subset = Subset(self.data, train_split.tolist())
        valid_subset = Subset(self.data, valid_split.tolist())
        test_subset = Subset(self.data, test_split.tolist())

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader, test_loader

