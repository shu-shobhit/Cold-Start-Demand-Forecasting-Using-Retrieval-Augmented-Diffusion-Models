"""Dataset utilities for RATD forecasting experiments.

This module defines the electricity forecasting dataset used by the project,
including normalization, train/validation/test splits, forecast masking, and
assembly of retrieved reference windows for retrieval-augmented diffusion.
"""

import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, Dataset

class Dataset_Electricity(Dataset):
    """Sliding-window electricity dataset tailored to RATD forecasting.

    Args:
        root_path: Directory containing the electricity CSV file.
        flag: Dataset split identifier, one of ``train``, ``val``, or ``test``.
        size: Sequence configuration ``[seq_len, label_len, pred_len, dim]``.
        features: Unused legacy argument retained for compatibility.
        data_path: CSV filename under ``root_path``.
        target: Unused legacy target name retained for compatibility.
        scale: Whether to standardize the data using train-split statistics.
        timeenc: Unused legacy flag retained for compatibility.
        freq: Unused legacy frequency string retained for compatibility.

    Returns:
        None: The constructor initializes the dataset in-place.
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='electricity.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        """Initialize the electricity forecasting dataset.

        Args:
            root_path: Directory containing the electricity CSV file.
            flag: Dataset split identifier.
            size: Sequence configuration ``[seq_len, label_len, pred_len, dim]``.
            features: Unused legacy argument retained for compatibility.
            data_path: CSV filename under ``root_path``.
            target: Unused legacy target name retained for compatibility.
            scale: Whether to standardize the data using train-split statistics.
            timeenc: Unused legacy flag retained for compatibility.
            freq: Unused legacy frequency string retained for compatibility.

        Returns:
            None: The constructor initializes the dataset in-place.
        """

        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.dim=size[3]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        """Load, split, normalize, and cache the electricity dataset.

        Args:
            None.

        Returns:
            None: The dataset state is populated on the instance.
        """

        df_raw = pd.read_csv(self.root_path+'/'+self.data_path, index_col='date', parse_dates=True)
        self.scaler = StandardScaler()

        df_raw = pd.DataFrame(df_raw)

        # The split points are adjusted by the forecast window so each slice can
        # still produce full history-plus-horizon training examples.
        num_train = int(len(df_raw) * 0.7)-self.pred_len-self.seq_len+1
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            # Normalization uses train-split statistics only, which prevents
            # leakage from validation and test windows.
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # Observation masks are dense in this forecasting setup because missing
        # values are created synthetically through ``gt_mask``.
        self.mask_data = np.ones_like(self.data_x)

        # Retrieved reference indices are precomputed offline by the TCN-based
        # retrieval script and loaded here for each training example.
        self.reference = torch.load('./dataset/TCN/ele_idx_list.pt')
        self.reference=torch.clamp(self.reference,min=0, max=17885)
        

    def __getitem__(self, index):
        """Assemble one forecasting sample and its retrieved references.

        Args:
            index: Start position of the sliding window inside the split.

        Returns:
            dict: Batch item containing data, masks, timepoints, and references.
        """

        s_begin = index
        s_end = s_begin + self.seq_len + self.pred_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # The current implementation concatenates the future segments from the
        # top-3 retrieved neighbor windows along the time axis.
        reference=np.zeros((3*self.pred_len, self.dim))
        reference[:self.pred_len, :]=self.data_x[int(self.reference[3*index])+self.seq_len:int(self.reference[3*index])+self.seq_len+self.pred_len]
        reference[self.pred_len:2*self.pred_len]=self.data_x[int(self.reference[3*index+1])+self.seq_len:int(self.reference[3*index+1])+self.seq_len+self.pred_len]
        reference[2*self.pred_len:3*self.pred_len]=self.data_x[int(self.reference[3*index+2])+self.seq_len:int(self.reference[3*index+2])+self.seq_len+self.pred_len]

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))
        

        # ``gt_mask`` controls which positions are revealed to the model. The
        # final prediction horizon is masked out so the model must forecast it.
        target_mask = self.mask_data[s_begin:s_end].copy()
        target_mask[-self.pred_len:] = 0. #pred mask for test pattern strategy
        s = {
            'observed_data':seq_x,
            'observed_mask': self.mask_data[s_begin:s_end],
            'gt_mask': target_mask,
            'timepoints': np.arange(self.seq_len + self.pred_len) * 1.0, 
            'feature_id': np.arange(370) * 1.0, 
            'reference': reference, 
        }

        return s

    def __len__(self):
        """Return the number of valid sliding windows in the split.

        Args:
            None.

        Returns:
            int: Number of available windows.
        """

        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Map normalized data back to the original scale.

        Args:
            data: Normalized array or tensor compatible with the scaler.

        Returns:
            numpy.ndarray: Data transformed back to the raw scale.
        """

        return self.scaler.inverse_transform(data)

def get_dataloader(device, batch_size=8):
    """Create train, validation, and test dataloaders for forecasting.

    Args:
        device: Unused legacy device argument retained for compatibility.
        batch_size: Mini-batch size for each dataloader.

    Returns:
        tuple: Train, validation, and test dataloaders.
    """

    # The dataset path is still hardcoded to the original environment from the
    # research snapshot. This function simply preserves that expectation.
    dataset = Dataset_Electricity(root_path='/data/0shared/liujingwei/dataset/ts2vec',flag='train',size=[96,0,168, 321])
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Dataset_Electricity(root_path='/data/0shared/liujingwei/dataset/ts2vec',flag='val',size=[96,0,168, 321])
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Dataset_Electricity(root_path='/data/0shared/liujingwei/dataset/ts2vec',flag='test',size=[96,0,168,321])
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=0)

    #scaler = torch.from_numpy(dataset.std_data).to(device).float()
    #mean_scaler = torch.from_numpy(dataset.mean_data).to(device).float()

    return train_loader, valid_loader, test_loader#, scaler, mean_scaler
