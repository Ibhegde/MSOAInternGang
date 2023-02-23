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
