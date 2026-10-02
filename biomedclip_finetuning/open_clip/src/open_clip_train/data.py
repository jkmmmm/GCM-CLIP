import logging
import os
import random
import time
from dataclasses import dataclass

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, disease_category, disease_location, config, sep="\t", tokenizer=None, allow_substitute=False):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)
        self.disease_category = df[disease_category].tolist()
        self.disease_location = df[disease_location].tolist()
        self.category = df["category"].tolist()
        self.location = df["location"].tolist()
        self.health_label_to_idx = {v: int(k)-1 for k, v in config['category'].items()}
        self.location_label_to_idx = {v: int(k)-1 for k, v in config['location'].items()}
        self.images = df[img_key].tolist()
        self.image_names = [os.path.basename(img_path) for img_path in self.images]
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

        self.allow_substitute = allow_substitute
        self._reported_bad = set()

        self.idx_to_health_label = {v: k for k, v in self.health_label_to_idx.items()}
        self.idx_to_location_label = {v: k for k, v in self.location_label_to_idx.items()}

    def __len__(self):
        return len(self.captions)

    def _load_image(self, idx):
        last_err = None
        for attempt in range(3):
            try:
                return self.transforms(Image.open(str(self.images[idx]))), None
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
        return None, last_err

    def __getitem__(self, idx):
        images, img_err = self._load_image(idx)
        if images is None:
            if not self.allow_substitute:
                raise img_err
            bad_path = str(self.images[idx])
            if bad_path not in self._reported_bad:
                self._reported_bad.add(bad_path)
                logging.warning(f'[CsvDataset] sample unreadable, substituted with a random sample: {bad_path} ({img_err})')
            for _ in range(10):
                j = random.randrange(len(self.captions))
                if j == idx:
                    continue
                images, _ = self._load_image(j)
                if images is not None:
                    idx = j
                    break
            else:
                raise img_err
        texts = self.tokenize([str(self.captions[idx])])[0]
        category = str(self.category[idx])
        location = str(self.location[idx])

        health_name = self.category[idx]
        health_label = torch.zeros(len(self.health_label_to_idx), dtype=torch.float)
        if health_name in self.health_label_to_idx:
            health_label[self.health_label_to_idx[health_name]] = 1.0

        location_name = self.location[idx]
        location_label = torch.zeros(len(self.location_label_to_idx), dtype=torch.float)
        if location_name in self.location_label_to_idx:
            location_label[self.location_label_to_idx[location_name]] = 1.0

        return images, texts, health_label, location_label


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None

    def set_epoch(self, epoch):
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_or_make_fixed_split(train_csv_path, val_csv_path, seed=42, val_size=1024):
    out_dir = os.path.dirname(os.path.abspath(train_csv_path))
    train_out = os.path.join(out_dir, f"fixed_train_s{seed}.csv")
    val_out = os.path.join(out_dir, f"fixed_val_s{seed}.csv")
    if os.path.exists(train_out) and os.path.exists(val_out):
        return train_out, val_out
    df_train = pd.read_csv(train_csv_path, low_memory=False)
    df_val = pd.read_csv(val_csv_path, low_memory=False)
    df = pd.concat([df_train, df_val], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = df.iloc[:val_size]
    train_df = df.iloc[val_size:]
    tmp_train, tmp_val = train_out + ".tmp", val_out + ".tmp"
    train_df.to_csv(tmp_train, index=False)
    val_df.to_csv(tmp_val, index=False)
    os.replace(tmp_train, train_out)
    os.replace(tmp_val, val_out)
    return train_out, val_out


def get_csv_dataset(args, preprocess_fn, is_train, config, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        disease_category=args.csv_disease_category,
        disease_location=args.csv_disease_location,
        config=config,
        sep=args.csv_separator,
        tokenizer=tokenizer,
        allow_substitute=is_train
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, config, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, config=config, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, config=config, tokenizer=tokenizer)

    return data
