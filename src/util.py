import concurrent.futures as cfu
import os
import random
import shutil

import numpy as np
import pandas as pd


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
        "POA_attribution_map": {"No": 0, "Yes": 1},
        "activity_category_map": {
            "Digital Media": 0,
            "Paid Social Media": 1,
            "Print": 2,
            "Out of Home Media": 3,
            "Out of Home": 4,
        },
        "activity_type_map": {
            "NonPartner.com": 0,
            "Member.com": 1,
            "Online Display": 2,
            "Magazine/Newspaper": 3,
            "Billboard/Transit": 4,
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
    max_image_count: int = 15,
    wait_to_complete: bool = True,
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
    with cfu.ThreadPoolExecutor() as executor:
        for img_lst in grp_src_df["image_name"].sample(sample_count):
            if max_image_count > 0:
                img_lst = random.choices(img_lst, k=max_image_count)
            for imgf in img_lst:
                src_file = os.path.join(src_path, imgf)
                copy_thr = executor.submit(shutil.copy, src=src_file, dst=dest_path)
                copy_lst[copy_thr] = dest_path
        if wait_to_complete:
            print("Waiting to complete copying....")
            for copy_thr in cfu.as_completed(copy_lst):
                copy_thr.result()
            print("Completed copying....")
