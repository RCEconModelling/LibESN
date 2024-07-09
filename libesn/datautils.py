import warnings
from typing import Type, Union

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class PanelDataset(Dataset):
    def __init__(
        self, 
        file: str,
        date_col: int=None,
        transform=None, 
        target_transform=None,
    ) -> None:
        # Ignore transform and target_transform 
        self.transform = None
        self.target_transform = None

        # Read data and prepare dataset
        raw_dataset = pd.read_csv(file)

        # Extract date index
        if not date_col is None:
            raw_dataset.set_index(raw_dataset.columns[date_col], inplace=True)
            self.index = raw_dataset.index
        else:
            self.index = pd.Index(np.arange(raw_dataset.shape[0]))
            warnings.warn("No date column specified, creating numeric index")

        if type(raw_dataset) is pd.DataFrame:
            self.data = raw_dataset.to_numpy()
        elif type(raw_dataset) is pd.Series:
            self.data = raw_dataset.to_numpy()[:,None]

    def __len__(self) -> int:
        # The panel dataset will consist of a single dataframe
        # with multiple time series and (possibly) a date index.
        # Its `Dataset` length will always be 1 (no batching).
        return 1
    
    def __getitem__(self, idx) -> torch.Tensor:
        # The dataset will always be loaded as a single dataframe
        # with multiple time series and (possibly) a date index.
        # The `idx` parameter is ignored.
        return torch.from_numpy(self.get_data())
    
    def get_data(self):
        return self.data
    
    def get_index(self, as_type: str = None) -> Union[list, np.ndarray, pd.Index]:
        if as_type is 'list':
            return self.index.to_list()
        elif as_type is 'numpy':
            return self.index.to_numpy()
        else:
            return self.index
