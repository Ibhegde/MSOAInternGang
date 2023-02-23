import argparse
import os
import shutil

from util import groupped_image_data, sample_images


def main():
    parser = argparse.ArgumentParser(
        prog="copy_sample_data.py",
        description="Copy image files samples to the local dir",
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
    grp_src_df = groupped_image_data(args.metadata)

    dest_path = args.dest_path
    # Remove old folder if exists
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    os.makedirs(dest_path, exist_ok=True)

    fetch_count = int(args.ref_count)

    sample_images(
        grp_src_df=grp_src_df,
        src_path=args.src_path,
        dest_path=args.dest_path,
        sample_count=fetch_count,
        max_image_count=-1,
    )


if __name__ == "__main__":
    main()
