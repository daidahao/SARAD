import csv
import os
from typing import Any, List, Optional, Tuple
import numpy as np

import torch
from torch.utils.data import random_split
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src import utils

from src.utils.logging_utils import log


class SMDBaseDataset(Dataset):
    """Base dataset within the SMD dataset."""

    def __init__(
            self, 
            window_size: int,
            data: np.ndarray,
            labels: np.ndarray,
            normalize: bool,
            forecast: bool,
        ) -> None:
        """Initialize a `SMDBaseDataset`.

        :param data_dir: The data directory.
        :param transform: The data transformations. Defaults to ``None``.
        """
        super().__init__()

        self.window_size = window_size
        self.data = data
        self.labels = labels
        self.normalize = normalize
        self.forecast = forecast

    def __len__(self) -> int:
        """Return the length of the dataset.

        :return: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset.

        :param idx: The index of the sample.
        :return: A sample from the dataset.
        """
        # get window
        if idx < self.window_size:
            start = self.data[[0]].repeat(self.window_size - idx - 1, axis=0)
            window = np.concatenate((start, self.data[:idx + 1]), axis=0)
        else:
            window = self.data[idx - self.window_size + 1:idx + 1]
        if self.normalize:
            window = window - window.mean(axis=0, keepdims=True)
            # window = window / window.std(axis=0, keepdims=True)
        if not self.forecast:
            return window, self.labels[idx]
        # get forecast window
        if idx > len(self) - self.window_size - 1:
            end = self.data[[-1]].repeat(self.window_size + idx - len(self) + 1, axis=0)
            forecast_window = np.concatenate((self.data[idx + 1:], end), axis=0)
        else:
            forecast_window = self.data[idx + 1: idx + 1 + self.window_size]
        return window, self.labels[idx], forecast_window
        
def parse_diagnosis(file_path):
    # Initialize an empty dictionary to store the parsed data
    parsed_data = {}

    # Open the file for reading
    with open(file_path, 'r') as file:
        # Iterate over each line in the file
        for line in file:
            # Strip whitespace and split the line on the colon to separate the range from the numbers
            range_part, numbers_part = line.strip().split(':')
            start, end = map(int, range_part.split('-'))  # Convert start and end of range to integers
            # Split the numbers part on commas to get individual numbers and convert them to integers
            numbers = [int(num)-1 for num in numbers_part.split(',')]
            # Store the range and numbers in the dictionary
            parsed_data[(start, end)] = numbers

    return parsed_data

class SMDSingleDataset(SMDBaseDataset):
    """Single dataset within the SMD dataset."""

    def __init__(
            self, 
            data_dir: str, 
            subset: str,
            machine_id: str,
            window_size: int,
            start: float,
            end: float,
            label_dir: Optional[str] = None,
        ) -> None:
        """Initialize a `SMDSingleDataset`.

        :param data_dir: The data directory.
        :param transform: The data transformations. Defaults to ``None``.
        """

        self.window_size = window_size

        # load data
        data = np.loadtxt(
            os.path.join(data_dir, subset, machine_id + '.txt'),
            delimiter=',',
        )
        
        data = data.astype(np.float32)
        start_idx = int(start * data.shape[0])
        end_idx = int(end * data.shape[0])
        data = data[start_idx:end_idx]

        # generate labels
        labels = np.zeros(data.shape[0], dtype=np.float32)
        if label_dir is not None:
            label_path = os.path.join(data_dir, label_dir, machine_id + '.txt')
            labels = np.loadtxt(label_path, delimiter=',', usecols=0)
            labels = labels[start_idx:end_idx]
        
        # todo: diagnosis labels
        if subset == 'test':
            diagnosis_data = parse_diagnosis(
                os.path.join(data_dir, 'interpretation_label', machine_id + '.txt'),)
        else:
            diagnosis_data = {}
        diagnosis = []
        for i in range(data.shape[0]):
            dv = []
            for (start, end), v in diagnosis_data.items():
                if i >= start and i < end:
                    dv = v
            diagnosis.append(dv)
        self.diagnosis = diagnosis
        
        # initialize dataset
        super().__init__(window_size, data, labels, normalize=False, forecast=False)


class SMDDataset(SMDBaseDataset):
    """Whole SMD dataset."""

    def __init__(
            self, 
            data_dir: str, 
            subset: str,
            window_size: int,
            post_scaler: Optional[Any] = None,
            post_scaler_class: Any = StandardScaler,
            num_datasets: int = 1000,
            start: float = 0.,
            end: float = 1.,
            label_dir: Optional[str] = None,
            forecast: bool = False,
        ) -> None:
        """Initialize a `SMDSingleDataset`.

        :param data_dir: The data directory.
        :param transform: The data transformations. Defaults to ``None``.
        """

        data_dir = os.path.join(data_dir, 'SMD')

        # load data
        datasets: List[Tuple[str, SMDSingleDataset]] = []

        filenames = os.listdir(os.path.join(data_dir, 'train'))
        filenames = [f for f in filenames if f.endswith('.txt')]
        for filename in sorted(filenames):
            machine_id = filename[:-len('.txt')]
            dataset = SMDSingleDataset(data_dir, subset, machine_id, \
                window_size, start, end, label_dir)
            datasets.append((machine_id, dataset))
            if len(datasets) == num_datasets:
                break
        
        data = np.concatenate([d.data for _, d in datasets])
        labels = np.concatenate([d.labels for _, d in datasets])
        self.diagnosis = [i for _, d in datasets for i in d.diagnosis]
        
        # perform post-scaling
        if post_scaler is None:
            self.post_scaler = post_scaler_class()
            self.post_scaler.fit(data)
        else:
            self.post_scaler = post_scaler
        data = self.post_scaler.transform(data)

        super().__init__(window_size, data, labels, \
                         normalize=False, forecast=forecast)


class SMDDataModule(LightningDataModule):
    """`LightningDataModule` for Anomaly Detection on SMD dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        input_size: int = 38,
        window_size: int = 10,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        percentile: float = 4.,
        sampling: float = 1.0,
        post_scaler_class: Any = StandardScaler,
        dataset: str = 'SMD',
        forecast: bool = False,
    ) -> None:
        """Initialize a `TSADDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param window_size: The window size. Defaults to `10`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.
        
        :param stage: The stage to setup. Defaults to `None`.
        """
        # load data from file
        self.data_train = SMDDataset(
            self.hparams.data_dir, 
            'train', 
            self.hparams.window_size, 
            post_scaler_class=self.hparams.post_scaler_class,
            start=0., end=0.8,
            forecast=self.hparams.forecast,
        )
        self.data_test = SMDDataset(
            self.hparams.data_dir, 
            'test', 
            self.hparams.window_size, 
            self.data_train.post_scaler,
            self.hparams.post_scaler_class,
            label_dir='test_label',
            forecast=self.hparams.forecast,
        )
        self.data_val = SMDDataset(
            self.hparams.data_dir, 
            'train', 
            self.hparams.window_size, 
            self.data_train.post_scaler,
            self.hparams.post_scaler_class,
            # num_datasets=5,
            start=0.8, end=1.,
            forecast=self.hparams.forecast,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader.

        :return: The train dataloader.
        """
        if self.hparams.sampling < 1:
            data_train, _ = random_split(self.data_train, \
                [self.hparams.sampling, 1.-self.hparams.sampling])
        else:
            data_train = self.data_train
        return DataLoader(
            data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader.

        :return: The validation dataloader.
        """
        if self.hparams.sampling < 1:
            data_val, _ = random_split(self.data_train, \
                    [self.hparams.sampling, 1.-self.hparams.sampling])
        else:
            data_val = self.data_train
        return DataLoader(
            data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.window_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Return the predict dataloader.

        :return: The predict dataloader.
        """
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`.

        :param stage: The stage being torn down. Defaults to `None`.
        """
        pass
    
