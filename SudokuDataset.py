import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class SudokuDataset(Dataset):
    pass
    def __init__(self, data: pd.DataFrame, prefixed=0.9):
        self.data = data.iloc[:int(data.shape[0]*prefixed)]

        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        matX = np.array(list(map(lambda x: int(x) if x != '.' else 0,
                                row.puzzle))).reshape(9, 9)
        matY = np.array(list(map(lambda x: int(x) if x != '.' else 0,
                                row.solution))).reshape(9, 9)

        matX = self.to_categorical(matX, 10).transpose(-1, 0, 1)
        matY = self.to_categorical(matY-1, 9).transpose(2, 0, 1)

        return matX, matY
    @classmethod
    def to_categorical(cls, y, num_classes, dtype='float32'):
        return (np.eye(num_classes)[y]).astype(dtype)

