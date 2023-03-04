import argparse
import os
import shutil
import pandas as pd
from util import groupped_image_data, sample_images, get_label_map
import math

# def copy_train_images(
#     grp_src_df: str, src_path: str, dest_path: str, sample_count: int
# ):
#     # Uniformy copy
#     label_sample_items = {}
#     for label_type, label_vals in get_label_map().items():
#         label_sample_list = []
#         for label_val in label_vals.keys():
#             labelled_dest = os.path.join(*[dest_path, label_type, "train", label_val])
#             sample_list = sample_images(
#                 grp_src_df=grp_src_df.loc[grp_src_df[label_type] == label_val],
#                 src_path=src_path,
#                 dest_path=labelled_dest,
#                 sample_count=sample_count,
#                 max_image_count=20,
#             )
#             label_sample_list.extend(
#                 [[ref_id, img_lst, label_val] for ref_id, img_lst in sample_list]
#             )
#         # print(label_sample_list)

#         sample_df = pd.DataFrame.from_records(
#             label_sample_list, columns=["ref_id", "image_names", label_type]
#         )
#         print(sample_df.head())
#         label_sample_items[label_type] = sample_df
#     return label_sample_items


def copy_sample_images(
    grp_src_df: str,
    src_path: str,
    dest_path: str,
    sample_count: int,
    sample_type: str = "test",
):
    if sample_type in ["train", "validation"]:
        # Uniformy copy
        label_sample_items = {}
        for label_type, label_vals in get_label_map().items():
            label_sample_list = []
            sample_path = os.path.join(*[dest_path, label_type, sample_type])
            label_count = math.ceil(sample_count / len(label_vals.keys()))
            for label_val in label_vals.keys():
                labelled_dest = os.path.join(sample_path, label_val)
                sample_list = sample_images(
                    grp_src_df=grp_src_df.loc[grp_src_df[label_type] == label_val],
                    src_path=src_path,
                    dest_path=labelled_dest,
                    sample_count=label_count,
                    max_image_count=20,
                )
                label_sample_list.extend(
                    [[ref_id, img_lst, label_val] for ref_id, img_lst in sample_list]
                )
            # print(label_sample_list)

            sample_df = pd.DataFrame.from_records(
                label_sample_list, columns=["ref_id", "image_name", label_type]
            )
            sample_df = sample_df.explode("image_name")

            sample_rec = os.path.join(sample_path, "record.csv")
            sample_df["file_name"] = sample_df["image_name"]
            sample_df["label"] = sample_df[label_type]
            sample_df[["image", "label"]].to_csv(sample_rec, index=False)
            label_sample_items[label_type] = sample_df
        return label_sample_items
    elif sample_type == "test":
        # Random copy
        label_sample_items = {}
        for label_type in get_label_map().keys():
            labelled_dest = os.path.join(*[dest_path, label_type, sample_type])
            sample_list = sample_images(
                grp_src_df=grp_src_df,
                src_path=src_path,
                dest_path=labelled_dest,
                sample_count=sample_count,
                category_col=label_type,
                max_image_count=20,
            )
            sample_refs = [ref_id for ref_id, img_lst in sample_list]
            sample_df = grp_src_df.loc[sample_refs][["image_name", label_type]]
            sample_df.reset_index()

            sample_df = sample_df.explode("image_name")
            sample_rec = os.path.join(labelled_dest, "record.csv")
            sample_df["file_name"] = sample_df["image_name"]
            sample_df["label"] = sample_df[label_type]
            sample_df[["image", "label"]].to_csv(sample_rec, index=False)

            label_sample_items[label_type] = sample_df
        return label_sample_items


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
        default="TRAIN_IMAGES",
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

    # sample_images(
    #     grp_src_df=grp_src_df,
    #     src_path=args.src_path,
    #     dest_path=args.dest_path,
    #     sample_count=fetch_count,
    #     max_image_count=-1,
    # )

    copy_sample_images(
        grp_src_df=grp_src_df,
        src_path=args.src_path,
        dest_path=args.dest_path,
        sample_count=fetch_count,
        sample_type="train",
    )

    copy_sample_images(
        grp_src_df=grp_src_df,
        src_path=args.src_path,
        dest_path=args.dest_path,
<<<<<<< HEAD
        sample_count=10,
=======
        sample_count=50,
>>>>>>> d574f56dd09a006c185e6b8351c999f0eebc606d
        sample_type="validation",
    )

    copy_sample_images(
        grp_src_df=grp_src_df,
        src_path=args.src_path,
        dest_path=args.dest_path,
<<<<<<< HEAD
        sample_count=10,
=======
        sample_count=50,
>>>>>>> d574f56dd09a006c185e6b8351c999f0eebc606d
        sample_type="test",
    )


if __name__ == "__main__":
    main()
