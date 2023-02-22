import os
import argparse
import pandas as pd
import numpy as np


def aggergate_data(meta_path: str):
    src_df = pd.read_csv(meta_path)

    # columns: image_name,ref_id, POA_attribution,activity_category,activity_type

    stat_df = src_df.groupby("ref_id").agg(
        {
            "image_name": len,
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


def main():
    parser = argparse.ArgumentParser(
        prog="copy_sample_data.py",
        description="Copy image files samples to the local dir",
    )

    parser.add_argument(
        "--path",
        action="store",
        default="TRAIN_IMAGES/",
        type=str,
        help="TRAIN data destination folder path",
    )
    parser.add_argument(
        "--metadata",
        action="store",
        default="data/TRAIN_images_metadata.csv",
        type=str,
        help="Test images metadata file path",
    )

    args = parser.parse_args()

    aggergate_data(args.metadata)


if __name__ == "__main__":
    main()
