import concurrent.futures as cfu
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
from torchvision import datasets


def groupped_image_data(metadata_file: str):
    src_df = pd.read_csv(metadata_file)
    stat_df = src_df.groupby("ref_id").agg(
        {
            "image_name": list,
            "POA_attribution": np.random.choice,
            "activity_category": np.random.choice,
            "activity_type": np.random.choice,
        }
    )

    return stat_df


def get_label_map():
    return {
        "POA_attribution": {"No": 0, "Yes": 1},
        "activity_category": {
            "Digital Media": 0,
            "Paid Social Media": 1,
            "Print": 2,
            "Out of Home Media": 3,
            "Out of Home": 4,
        },
        "activity_type": {
            "NonPartner.com": 0,
            "Member.com": 1,
            "Online Display": 2,
            "Magazine-Newspaper": 3,
            "Billboard-Transit": 4,
            "Collateral": 5,
            "Misc": 6,
            "IndustryPartner.com": 7,
        },
    }


def sample_images(
    grp_src_df: pd.DataFrame,
    src_path: str,
    dest_path: str,
    sample_count: int,
    category_col: str = None,
    max_image_count: int = 20,
):
    """Give random sameples from the input data groupped by reference id

    Args:
        grp_src_df (pd.DataFrame): _description_
        src_path (str): _description_
        dest_path (str): _description_
        sample_count (int): _description_
        max_image_count (int, optional): _description_. Defaults to 15. Set -1 to select all images.
        wait_to_complete (bool, optional): wait till the copy threads complete execution
    """
    copy_lst = {}
    grp_src_df = groupped_image_data("data/TRAIN_images_metadata.csv")
    with cfu.ThreadPoolExecutor() as executor:
        if sample_count is None:
            sample_count = len(grp_src_df["image_name"])

        sample_count = min(sample_count, len(grp_src_df["image_name"]))
        sample_rows = grp_src_df["image_name"].sample(sample_count)
        labelled_dest = dest_path
        os.makedirs(labelled_dest, exist_ok=True)

        for smp_ref, img_lst in sample_rows.items():
            # Add label to the dest path if label is provided
            if category_col is not None:
                cat_label = grp_src_df.loc[smp_ref][category_col]
                labelled_dest = os.path.join(dest_path, cat_label)
                os.makedirs(labelled_dest, exist_ok=True)

            if max_image_count > 0 and len(img_lst) > max_image_count:
                img_lst = random.sample(img_lst, k=max_image_count)
            for imgf in img_lst:
                src_file = os.path.join(src_path, imgf)
                dst_file = os.path.join(labelled_dest, imgf)
                if os.name == "nt":
                    copy_thr = executor.submit(shutil.copy, src=src_file, dst=dst_file)
                else:
                    copy_thr = executor.submit(os.symlink, src=src_file, dst=dst_file)
                copy_lst[copy_thr] = (smp_ref, img_lst)

        print("Waiting to complete copying....")
        for copy_thr in cfu.as_completed(copy_lst):
            try:
                copy_thr.result()
            except Exception as e:
                print(e)

        print("Completed copying....")
    return list(copy_lst.values())


def get_data_set(root_dir: str, sample_type: str, transform=None):
    data_dir = os.path.join(root_dir, sample_type)
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader
