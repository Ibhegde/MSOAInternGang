import os

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

from src.util import get_label_map


class TrainModel:
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        # return accuracy_score(y_true = labels, y_pred = predictions)
        return {"f1": float(f1_score(y_true=labels, y_pred=predictions))}

    def collate_fn(self, batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch]),
        }

    def transform_image(self, image_files):
        inputs = self.feature_ext(
            [x.convert("RGB") for x in image_files["image"]], return_tensors="pt"
        )
        inputs["labels"] = image_files["label"]
        return inputs

    def train(self):
        train_results = self.trainer.train()
        self.trainer.save_model()
        self.trainer.log_metrics("train", train_results.metrics)
        self.trainer.save_metrics("train", train_results.metrics)
        self.trainer.save_state()
        return train_results

    def test(self):
        metrics = self.trainer.evaluate(self.prep_ds["test"])
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        return metrics

    def __init__(
        self,
        model_name: str,
        label_col: str,
        image_dir: str = "TRAIN_IMAGES/",
        output_dir: str = "vit-base-aie",
    ) -> None:
        self.model_name = model_name
        self.feature_ext = ViTImageProcessor.from_pretrained(
            self.model_name, proxies={"https": "proxy-ir.intel.com:912"}
        )

        self.label_col = label_col
        self.labels_lst = get_label_map()[self.label_col]
        self.output_dir = os.path.join(output_dir, self.label_col)

        # To allow loading large images
        Image.MAX_IMAGE_PIXELS = None
        dataset = load_dataset(
            "imagefolder",
            data_dir=os.path.join(image_dir, self.label_col),
            drop_labels=False,
        )
        self.prep_ds = dataset.with_transform(self.transform_image)

        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels_lst),
            id2label={v: k for k, v in self.labels_lst.items()},
            label2id=self.labels_lst,
            proxies={"https": "proxy-ir.intel.com:912"},
        )

        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=16,
            evaluation_strategy="steps",
            num_train_epochs=10,
            fp16=True,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=1e-6,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="tensorboard",
            load_best_model_at_end=True,
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.collate_fn,
            compute_metrics=self.compute_metrics,
            train_dataset=self.prep_ds["train"],
            eval_dataset=self.prep_ds["validation"],
            tokenizer=self.feature_extractor,
        )


def main():
    # TODO: take arguments in commandline#
    model_name = "google/vit-base-patch16-224-in21k"

    # for label in get_label_map():
    #     label_col = label

    label_col = "POA_attribution"

    tm = TrainModel(model=model_name, label_col=label_col, output_dir="test-vit-base")
    tm.train()
    tm.test()


if __name__ == "__main__":
    main()
