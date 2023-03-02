import pandas as pd
import os
import argparse
from src.baseline_pred import max_predict
from src.finetune_vit import TrainModel
from src.util import get_label_map
import numpy as np

parser = argparse.ArgumentParser(
    prog="predict.py", description="Prediction python module for Hackathon"
)
parser.add_argument(
    "--path",
    action="store",
    default="/data/",
    required=True,
    type=str,
    help="TEST data folder path",
)
parser.add_argument(
    "--metadata",
    action="store",
    default="TEST_images_metadata.csv",
    required=True,
    type=str,
    help="Test images metadata file path",
)

args = parser.parse_args()

# all images are available here.
# all_images = os.listdir(args.path)
# print(len(all_images))

# Do your magic here....
# e.g. loading pre-processing functions, dataloaders, helper_functions & models

pred_df = pd.read_csv(args.metadata)
# pred_df = pred_df.loc[pred_df["ref_id"] == "00113332-001"]
pred_df = pred_df.loc[:5]

print(len(pred_df))
label_types = get_label_map()
for label in label_types:
    tm = TrainModel(
        model_name=os.path.join("vit-base-aie-15k", label),
        label_col=label,
        output_dir=None,
        image_dir=None,
    )
    # pred_df[label] = pred_df["image_name"].map(
    #     lambda img: tm.predict(img_path=os.path.join(args.path, img))
    # )
    pred_df[label] = tm.predict_batch(img_df=pred_df, img_path=args.path)

pred_df = pred_df.groupby("ref_id").agg(
    {
        "image_name": list,
        "POA_attribution": np.max,
        "activity_category": lambda el: max(set(el.tolist()), key=el.tolist().count),
        "activity_type": lambda el: max(set(el.tolist()), key=el.tolist().count),
    }
)

pred_df = pred_df.explode("image_name")

# df = max_predict(args.metadata)
# Get predictions at ref_id level.

# Create dataframe with the below specified template/ the sample submission file format mentioned on the Platform.
# There are 3 tasks, if you're not solving any task, please exclude corresponding column while generating below dataframe.

# ref_id,       POA_attribution, activity_category, activity_type
# 00113332-001,         0,               1,                3
# 00147376-001,         1,               3,                6
# .
# .
# .
# ----------------------------------------------------------------
# DON'T CHANGE THE BELOW FILE NAME, ELSE, INFERENCE PROCESS FAILS.
# ----------------------------------------------------------------
# save the dataframe as ===== 'predictions.csv' =====
pred_df.to_csv("predictions.csv", index=False)

# Exit.


#### To TEST Locally ####
# python predict.py --path path/to/your/local/test/folder  --metadata path/to/TEST_images_metadata.csv
