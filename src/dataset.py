from torch.utils.data import Dataset, DataLoader, SequentialSampler
from PIL import Image
import os
import numpy as np
import torch


class AIECVDataSet(Dataset):
    """AIE CV dataset."""

    def __init__(self, stat_df, root_dir, transform=None, label_col=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.stat_df = stat_df
        self.root_dir = root_dir
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.stat_df)  # return length (numer of rows) of the dataframe

    def __getitem__(self, idx):
        img_pil = Image.open(
            os.path.join(self.root_dir, self.stat_df.iloc[idx, 0])
        ).convert("RGB")
        pixel_vales = self.transform(img_pil)
        if self.label_col:
            labels = self.stat_df.iloc[idx, self.label_col]
            return pixel_vales, labels

        return pixel_vales

    def collate_fn_ul(self, sample_batch):
        return torch.stack([x for x in sample_batch])

    def get_unlabelled_data(self):
        sampler = SequentialSampler(self)
        return DataLoader(
            self, batch_size=32, collate_fn=self.collate_fn_ul, sampler=sampler
        )
