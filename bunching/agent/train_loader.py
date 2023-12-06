import pandas as pd
import torch
from torch.utils.data import Dataset


class Step_Dataset(Dataset):
    def __init__(self, stats, actis, retus):
        self.__stats = torch.tensor(stats, dtype=torch.float32).reshape(-1, 2)
        self.__actis = torch.tensor(actis, dtype=torch.int32).reshape(-1, 1)
        self.__retus = torch.tensor(retus, dtype=torch.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.__stats)

    def __getitem__(self, idx):
        s = self.__stats[idx]
        a = self.__actis[idx]
        q = self.__retus[idx]
        return s, a, q
