import concurrent.futures as cfu
import os
import sys

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image, ImageFile
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomAffine,
    RandomRotation,
    RandomPerspective,
    RandomApply,
    ColorJitter,
    Resize,
    ToTensor,
)
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

from .dataset import AIECVDataSet
from .util import get_label_map
from .custom_classifier import SimpleFCs


class ProcessImage:
    def __init__(self, model_name) -> None:
        # To allow loading large images
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.feature_ext = ViTImageProcessor.from_pretrained(
            model_name, proxies={"https": "proxy-ir.intel.com:912"}
        )

        self.size = self.feature_ext.size["height"]
        self.normalise = Normalize(
            mean=self.feature_ext.image_mean, std=self.feature_ext.image_std
        )

        self.preprocess_train = Compose(
            [
                RandomResizedCrop(self.size),
                RandomApply(
                    [
                        RandomHorizontalFlip(),
                        RandomAffine((30, 120)),
                        RandomPerspective(),
                        RandomRotation((30, 120)),
                        ColorJitter(),
                    ]
                ),
                ToTensor(),
                self.normalise,
            ]
        )
        self.preprocess_val = Compose(
            [
                Resize(self.size),
                CenterCrop(self.size),
                ToTensor(),
                self.normalise,
            ]
        )


class TrainModel:
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        # return accuracy_score(y_true = labels, y_pred = predictions)
        return {
            "f1": float(f1_score(y_true=labels, y_pred=predictions, average="weighted"))
        }

    def collate_fn(self, batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["label"] for x in batch]),
        }

    def train_transform_image(self, image_files):
        image_files["pixel_values"] = [
            self.pi.preprocess_train(pi_img.convert("RGB"))
            for pi_img in image_files["image"]
        ]
        return image_files

    def val_transform_image(self, image_files):
        image_files["pixel_values"] = [
            self.pi.preprocess_val(pi_img.convert("RGB"))
            for pi_img in image_files["image"]
        ]
        return image_files

    def train(self):
        if next(self.model.parameters()).is_cuda:
            print("Running on GPU!!")
        train_results = self.trainer.train()
        self.trainer.save_model()
        self.trainer.log_metrics("train", train_results.metrics)
        self.trainer.save_metrics("train", train_results.metrics)
        self.trainer.save_state()
        return train_results

    def test(self):
        metrics = self.trainer.evaluate(self.test_ds)
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        return metrics

    def predict_batch(self, pred_ds):
        pred_ds.set_transform(self.pi.preprocess_val)
        pred_dataloader = pred_ds.get_unlabelled_data()
        pred_cl = []
        pred_wt = []

        for batch_input in pred_dataloader:
            batch_input = batch_input.to(self.device)
            outputs = self.model(batch_input)
            logits = outputs.logits
            if self.device == "cuda":
                logits = logits.to("cpu")
            # predicted_class = torch.argmax(logits, -1).numpy()
            pred_wtb, pred_clb = torch.max(logits, -1)
            pred_cl.extend(pred_clb.detach().numpy())
            pred_wt.extend(pred_wtb.detach().numpy())
        # print(pred_wt)
        return pred_cl, pred_wt

    def __init__(
        self,
        model_name: str,
        label_col: str,
        image_dir: str = "/home/jovyan/team3/MSOAInternGang/TRAIN_IMAGES/",
        output_dir: str = "vit-base-aie-test",
    ) -> None:
        if len(sys.argv) == 2:
            print("Using GPUs 1 and 2")
            os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
        self.device = "cuda"
        if not torch.cuda.is_available():
            self.device = "cpu"

        self.model_name = model_name

        self.pi = ProcessImage(model_name=model_name)

        self.label_col = label_col
        self.labels_lst = get_label_map()[self.label_col]

        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.labels_lst),
            id2label={v: k for k, v in self.labels_lst.items()},
            label2id=self.labels_lst,
            proxies={"https": "proxy-ir.intel.com:912"},
        )

        self.model.classifier = SimpleFCs(
            hidden_size=self.model.config.hidden_size, num_labels=len(self.labels_lst)
        )
        self.model = self.model.to(self.device)

        # freeze params of pretrained model
        for param in self.model.vit.parameters():
            param.requires_grad = False

        if image_dir is not None and output_dir is not None:
            self.output_dir = os.path.join(output_dir, self.label_col)
            print("data dir %s" % (os.path.join(image_dir, self.label_col)))
            dataset = load_dataset(
                "imagefolder",
                data_dir=os.path.join(image_dir, self.label_col),
                drop_labels=False,
            )
            self.train_ds = dataset["train"].with_transform(self.train_transform_image)
            self.val_ds = dataset["validation"].with_transform(self.val_transform_image)
            self.test_ds = dataset["train"].with_transform(self.val_transform_image)

            print("Train set: %d rows" % (self.train_ds.num_rows))
            print("Validation set: %d rows" % (self.train_ds.num_rows))
            self.training_args = TrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=32,
                evaluation_strategy="steps",
                num_train_epochs=10,
                # fp16=True,
                save_steps=200,
                eval_steps=200,
                logging_steps=20,
                learning_rate=1e-3,
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
                train_dataset=self.train_ds,
                eval_dataset=self.val_ds,
                tokenizer=self.pi.feature_ext,
            )


def train_model(model_name, label_col):
    tm = TrainModel(
        model_name=model_name,
        label_col=label_col,
        output_dir="custom-levit-3k-lr1e-3d88",
        image_dir="/mnt/hdd/fab_data/aie_hackathon/TRAIN_IMAGES_best",
    )
    trm = tm.train()
    tstm = tm.test()
    return (trm, tstm)


def main():
    # TODO: take arguments in commandline#
    model_name = "google/vit-base-patch16-224-in21k"
    is_custom = False

    trainers = {}
    results = {}
    with cfu.ThreadPoolExecutor() as executor:
        for label in list(get_label_map().keys()):
            label_col = label
            print(
                "************************ label_col: %s *****************************"
                % label_col
            )
            train_model_name = model_name
            if is_custom:
                train_model_name = os.path.join(model_name, label_col)
            tr_exe = executor.submit(train_model, train_model_name, label_col)
            trainers[tr_exe] = label_col
    for tr_exe in cfu.as_completed(trainers):
        label_col = trainers[tr_exe]
        results[label_col] = tr_exe.result()

    # results = {}
    # for label in list(get_label_map().keys()):
    #     label_col = label
    #     print(
    #         "************************ label_col: %s *****************************"
    #         % label_col
    #     )
    #     train_model_name = model_name
    #     if is_custom:
    #         train_model_name = os.path.join(model_name, label_col)
    #     trm, tstm = train_model(train_model_name, label_col)
    #     results[label_col] = (trm, tstm)

    for label in results:
        trm, tstm = results[label]
        print("Label type: %s" % label)
        print(trm)
        print(tstm)
        print("")


if __name__ == "__main__":
    main()
