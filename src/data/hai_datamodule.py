import csv
from datetime import datetime
import os
from typing import Any, List, Optional, Tuple
import numpy as np

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import random_split
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src import utils
from src.data.components.hai import attacks

from src.utils.logging_utils import log


class HAIBaseDataset(Dataset):
    """Base dataset within the HAI dataset."""

    def __init__(
            self, 
            window_size: int,
            data: np.ndarray,
            labels: np.ndarray,
            normalize: bool,
            forecast: bool,
        ) -> None:
        """Initialize a `HAISingleDataset`.

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

class HAISingleDataset(HAIBaseDataset):
    """Single dataset within the HAI dataset."""

    def __init__(
            self, 
            data_dir: str, 
            filename: str,
            input_size: int,
            window_size: int,
            start: float,
            end: float,
        ) -> None:
        """Initialize a `HAISingleDataset`.

        :param data_dir: The data directory.
        :param transform: The data transformations. Defaults to ``None``.
        """

        self.window_size = window_size

        # load data
        data_path = os.path.join(data_dir, filename)
        data = np.loadtxt(
            data_path,
            delimiter=',', 
            dtype=np.float32, 
            skiprows=1, 
            usecols=range(1, input_size + 2),
        )

        # load ts
        ts = np.loadtxt(data_path, delimiter=',', skiprows=1, \
                        usecols=0, dtype=str)
        
        # load columns
        columns = np.loadtxt(data_path, delimiter=',', \
                    usecols=range(1, input_size + 1), \
                    dtype=str, max_rows=1)
        columns = list(map(str.strip, columns))

        # parse each attack's points
        for attack in attacks:
            attack['columns'] = []
            for point in attack['points']:
                if point not in columns:
                    log.debug(f'{attack["no"]}-{point} not in columns.')
                else:
                    index = columns.index(point)
                    attack['columns'].append(index)
        
        start_idx = int(start * data.shape[0])
        end_idx = int(end * data.shape[0])
        data = data[start_idx:end_idx]

        # generate labels
        labels = data[:, -1]
        data = data[:, :-1]

        # generate diagnosis
        self.diagnosis = []
        for t in ts:
            d = []
            t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            for attack in attacks:
                if attack['start_time_dt'] <= t < attack['end_time_dt']:
                    d = attack['columns']
            self.diagnosis.append(d)

        # initialize dataset
        super().__init__(window_size, data, labels, \
                         normalize=False, forecast=False)


class HAIDataset(HAIBaseDataset):
    """Whole HAI dataset."""

    def __init__(
            self, 
            data_dir: str, 
            subset: str,
            input_size: int,
            window_size: int,
            post_scaler: Optional[Any] = None,
            post_scaler_class: Any = StandardScaler,
            num_datasets: int = 1000,
            start: float = 0.,
            end: float = 1.,
            forecast: bool = False,
        ) -> None:
        """Initialize a `HAISingleDataset`.

        :param data_dir: The data directory.
        :param subset: The subset of the dataset.
        :param input_size: The input size.
        :param window_size: The window size.
        :param post_scaler: The post-scaler. Defaults to ``None``.
        :param post_scaler_class: The post-scaler class. Defaults to ``StandardScaler``.
        :param num_datasets: The number of datasets to load. Defaults to ``1000``.
        :param start: The start of the dataset. Defaults to ``0.``.
        :param end: The end of the dataset. Defaults to ``1.``.
        """

        data_dir = os.path.join(data_dir, 'HAI', 'hai-21.03')

        # load data
        datasets: List[Tuple[str, HAISingleDataset]] = []

        filenames = [f for f in os.listdir(data_dir) \
                     if f.startswith(subset) and f.endswith('.csv')]
        for filename in sorted(filenames):
            dataset = HAISingleDataset(data_dir, filename, input_size, \
                window_size, start, end)
            datasets.append((filename, dataset))
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


class HAIDataModule(LightningDataModule):
    """`LightningDataModule` for Anomaly Detection on HAI dataset.

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
        input_size: int = 79,
        window_size: int = 10,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        percentile: float = 4.,
        sampling: float = 1.0,
        post_scaler_class: Any = StandardScaler,
        dataset: str = 'HAI',
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
        data_train = HAIDataset(
            self.hparams.data_dir, 
            'train', 
            self.hparams.input_size,
            self.hparams.window_size, 
            post_scaler_class=self.hparams.post_scaler_class,
            start=0., end=1.,
            forecast=self.hparams.forecast,
        )
        self.data_test = HAIDataset(
            self.hparams.data_dir, 
            'test', 
            self.hparams.input_size,
            self.hparams.window_size, 
            data_train.post_scaler,
            self.hparams.post_scaler_class,
            forecast=self.hparams.forecast,
        )
        self.data_train, self.data_val = train_test_split(data_train, train_size=0.8, shuffle=False)
        self.data_train_org = data_train
        
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
    
