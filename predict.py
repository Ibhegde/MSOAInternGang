import pandas as pd
import os
import argparse

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
all_images = os.listdir(args.path)

# Do your magic here....
# e.g. loading pre-processing functions, dataloaders, helper_functions & models

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
# df.to_csv("predictions.csv", index=False)

# Exit.


#### To TEST Locally ####
# python predict.py --path path/to/your/local/test/folder  --metadata path/to/TEST_images_metadata.csv
