import os
import argparse
import pandas as pd
import shutil
import concurrent.futures as cfu

parser = argparse.ArgumentParser(
    prog="copy_sample_data.py", description="Copy image files samples to the local dir"
)

parser.add_argument(
    "--src-path",
    action="store",
    required=True,
    type=str,
    help="TRAIN data source folder path",
)

parser.add_argument(
    "--dest-path",
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

parser.add_argument(
    "--ref-count",
    action="store",
    default="10",
    type=str,
    help="Test images metadata file path",
)

args = parser.parse_args()
src_df = pd.read_csv(args.metadata)
grp_src_df = (
    src_df[["image_name", "ref_id"]].groupby("ref_id")["image_name"].apply(list)
)

dest_path = args.dest_path
# Remove old folder if exists
if os.path.exists(dest_path):
    shutil.rmtree(dest_path)
os.makedirs(dest_path, exist_ok=True)

fetch_count = int(args.ref_count)

with cfu.ThreadPoolExecutor() as executor:
    for item in grp_src_df.head(fetch_count):
        for imgf in item:
            src_file = os.path.join(args.src_path, imgf)
            executor.submit(shutil.copy, src=src_file, dst=args.dest_path)
