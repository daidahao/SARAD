from datetime import datetime
import os
import time
from typing import Any, Optional, Tuple
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from xlsx2csv import Xlsx2csv

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src import utils
from src.data.components import swat

from src.utils.logging_utils import log

class SWaTDataset(Dataset):
    """SWaT Dataset."""

    def __init__(
            self, 
            data_path: str, 
            attacks: list[dict],
            input_size: int,
            window_size: int,
            post_scaler: Optional[TransformerMixin] = None,
            post_scaler_class: Any = StandardScaler,
            max_rows: Optional[int] = None,
            forecast: bool = False,
        ) -> None:
        """Initialize a `SWaTDataset`.

        :param data_path: The path to the data.
        :param attacks: SWaT attacks, including start and end time.
        :param input_size: The number of input features.
        :param window_size: The window size.
        :param post_scaler: The post-scaler. Defaults to `None`.
        :param post_scaler_class: The post-scaler class. Defaults to `StandardScaler`.
        :param max_rows: The maximum number of rows to load. Defaults to `None`.
        """
        super().__init__()

        self.window_size = window_size
        self.forecast = forecast

        # load data
        ts = np.loadtxt(data_path, delimiter=',', skiprows=2, \
                        usecols=0, dtype=str, max_rows=max_rows)
        self.data = np.loadtxt(data_path, delimiter=',', skiprows=2, \
                               usecols=range(1, input_size + 1), \
                                 max_rows=max_rows)
        ts, self.data = ts[:max_rows], self.data[:max_rows]
        self.data = self.data.astype(np.float32)

        # load columns
        columns = np.loadtxt(data_path, delimiter=',', \
                    usecols=range(1, input_size + 1), \
                    dtype=str, max_rows=1)
        columns = list(map(str.strip, columns))

        # generate labels
        self.labels = np.zeros(self.data.shape[0], dtype=np.float32)
        self.diagnosis = []
        for i, t in enumerate(ts):
            t = t.strip()
            t = datetime.strptime(t, '%d/%m/%Y %I:%M:%S %p')
            d = []
            for attack in attacks:
                if attack['start_time_dt'] <= t <= attack['end_time_dt']:
                    self.labels[i] = 1.
                    points = map(lambda p: p.replace('-', ''), attack['points'])
                    d = [ columns.index(p) \
                         for p in points if p in columns]
                    break
            self.diagnosis.append(d)
        
        # perform post-scaling
        if post_scaler is None:
            # self.post_scaler = MinMaxScaler()
            self.post_scaler = post_scaler_class()
            self.post_scaler.fit(self.data)
        else:
            self.post_scaler = post_scaler

        self.data = self.post_scaler.transform(self.data)


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
        if not self.forecast:
            return window, self.labels[idx]

        # get forecast window
        if idx > len(self) - self.window_size - 1:
            end = self.data[[-1]].repeat(self.window_size + idx - len(self) + 1, axis=0)
            forecast_window = np.concatenate((self.data[idx + 1:], end), axis=0)
        else:
            forecast_window = self.data[idx + 1: idx + 1 + self.window_size]
        return window, self.labels[idx], forecast_window


class SWaTDataModule(LightningDataModule):
    """`LightningDataModule` for Anomaly Detection on SWaT dataset.

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
        input_size: int = 51,
        window_size: int = 10,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        percentile: float = 0.1,
        post_scaler_class: Any = StandardScaler,
        dataset: str = 'SWaT',
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

        self.normal_file = 'SWaT_Dataset_Normal_v1'
        self.attack_file = 'SWaT_Dataset_Attack_v0'

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
        data_dir = os.path.join(self.hparams.data_dir, 'SWaT')
        log.info(f"Preparing data in {data_dir}...")
        t = time.time()
        for file in [self.normal_file, self.attack_file]:
            inpath = os.path.join(data_dir, f'{file}.xlsx')
            outpath = os.path.join(data_dir, f'{file}.csv')
            if os.path.exists(outpath):
                continue
            Xlsx2csv(inpath, outputencoding="utf-8").convert(outpath)
        log.info(f"Data prepared in {time.time() - t:.2f}s.")
        

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.
        
        :param stage: The stage to setup. Defaults to `None`.
        """
        if self.data_train and self.data_val and self.data_test:
            # don't do anything if setup has already been called
            return
        log.info(f"Setting up data for stage {stage}...")
        t = time.time()
        # load data from file
        data_dir = os.path.join(self.hparams.data_dir, 'SWaT')
        data_train = SWaTDataset(
            os.path.join(data_dir, f'{self.normal_file}.csv'),
            [],
            self.hparams.input_size,
            self.hparams.window_size,
            post_scaler_class=self.hparams.post_scaler_class,
            forecast=self.hparams.forecast,
        )
        self.data_test = SWaTDataset(
            os.path.join(data_dir, f'{self.attack_file}.csv'),
            swat.attacks,
            self.hparams.input_size,
            self.hparams.window_size,
            # self.data_train.post_scaler,
            post_scaler_class=self.hparams.post_scaler_class,
            forecast=self.hparams.forecast,
        )
        self.data_train, self.data_val = train_test_split(data_train, train_size=0.8, shuffle=False)
        log.info(f"Data loaded in {time.time() - t:.2f}s.")
        log.info(f"Data size: {len(self.data_train)}, " \
                    f"{len(self.data_val)}, {len(self.data_test)}")

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            self.data_val,
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
    
