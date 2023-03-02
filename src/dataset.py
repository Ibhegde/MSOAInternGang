from torch.utils.data import Dataset, DataLoader
from PIL import Image
import tensorflow as tf
import pandas as pd
import os
import numpy as np
from util import get_label_map


class AIECVDataSet(Dataset):
    """AIE CV dataset."""

    def __init__(self, csv_file, root_dir, label_col, transform):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.stat_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_col = label_col

    def __len__(self):
        return len(self.stat_df)  # return length (numer of rows) of the dataframe

    def __getitem__(self, idx):
        img_pil = Image.open(
            os.path.join(self.root_dir, self.stat_df.iloc[idx, "image_name"])
        ).convert("RGB")
        pixel_vales = self.transform(img_pil, return_tensors="pt")
        labels = self.stat_df.iloc[idx, self.label_col]
        label_val = get_label_map()[self.label_col][labels]
        return pixel_vales, labels
