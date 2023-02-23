import pandas as pd
from .util import get_label_map


def max_predict(metadata_file: str):

    src_df = pd.read_csv(metadata_file)

    # columns: image_name,ref_id, POA_attribution,activity_category,activity_type
    stat_df = src_df.groupby("ref_id").agg(
        {
            "image_name": len,
        }
    )
    agg_data = pd.DataFrame(
        columns=[
            "ref_id",
            "POA_attribution",
            "activity_category",
            "activity_type",
        ]
    )

    agg_data["ref_id"] = stat_df.index
    label_map = get_label_map()
    print(label_map)
    base_poa = label_map["POA_attribution_map"]["Yes"]
    base_ac = label_map["activity_type_map"]["NonPartner.com"]
    base_at = label_map["activity_category_map"]["Digital Media"]

    agg_data["POA_attribution"].fillna(base_poa, inplace=True)
    agg_data["activity_category"].fillna(base_ac, inplace=True)
    agg_data["activity_type"].fillna(base_at, inplace=True)

    return agg_data
