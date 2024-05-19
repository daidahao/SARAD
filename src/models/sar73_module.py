from typing import Any, Dict, Tuple
import numpy as np

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

from src import utils
from src.models.components.sar73 import SAR73
from src.utils.record import save_scores
from src.utils.vus import calculate_all_metrics

from src.utils.logging_utils import log


class SAR73Module(LightningModule):
    """SAR73 module for anomaly detection.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
            output_dir: str,
            dataset: str = 'MSL',
    ) -> None:
        """Initialize a `TSADModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model.
        """
        super().__init__()

        # nvidia gpu optimisation
        torch.set_float32_matmul_precision('medium')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore="net")

        self.net: SAR73 = net

        # loss function
        self.criterion = torch.nn.MSELoss(reduction="none")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        return self.net(x)
    
    
    def model_step(
            self, 
            batch: Tuple[torch.Tensor, torch.Tensor],
            mode: str = "train"
    ) -> torch.Tensor:
        """Perform a single model step.

        :param batch: A tuple containing the input and target tensors.
        :return: A tensor of predictions.
        """

        output = batch
        x, y = output

        # forward pass
        x_hat, s = self(x)

        # calculate losses & scores
        # shape: [batch_size]
        scores = recon_losses = self.criterion(x_hat, x).mean(dim=(1, 2))


        return {
            "recon_losses": recon_losses,
            "y": y,
            "scores": scores,
        }

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch begins."""
        self.train_loss.reset()


    def training_step(
            self, 
            batch: Tuple[torch.Tensor, torch.Tensor], 
            batch_idx: int,
    ) -> Dict[str, Any]:
        """Perform a single training step.

        :param batch: A tuple containing the input and target tensors.
        :param batch_idx: The index of the batch.
        :return: A dictionary containing the loss and metrics.
        """

        output = self.model_step(batch)

        # calculate loss
        loss = output["recon_losses"].mean()
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when a validation epoch begins."""
        self.val_loss.reset()

    def validation_step(
            self, 
            batch: Tuple[torch.Tensor, torch.Tensor], 
            batch_idx: int
    ) -> Dict[str, Any]:
        """Perform a single validation step.

        :param batch: A tuple containing the input and target tensors.
        :param batch_idx: The index of the batch.
        """
        # output = self.model_step(batch)
        # loss = output["recon_losses"].mean()
        # self.val_loss(loss)
        # self.log("val/loss", self.val_loss, \
        #          on_step=False, on_epoch=True, prog_bar=True)


    def on_test_start(self) -> None:
        """Lightning hook that is called when testing begins."""
        self.test_ys = []
        self.test_scores = []

    def test_step(
            self, 
            batch: Tuple[torch.Tensor, torch.Tensor], 
            batch_idx: int
    ) -> None:
        """Perform a single test step.

        :param batch: A tuple containing the input and target tensors.
        :param batch_idx: The index of the batch.
        """
        output = self.model_step(batch, mode="test")
        
        self.test_ys.append(output["y"].cpu().numpy())
        self.test_scores.append(output["scores"].cpu().numpy())
    
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a testing epoch ends."""
        # get all scores and labels
        y = np.concatenate(self.test_ys)
        scores = np.concatenate(self.test_scores)

        save_scores(y, scores, 0.0, 'SAR73', \
                    self.hparams.dataset, 'logs/models/')

        self.log("test/scores", np.mean(scores), on_epoch=True, prog_bar=True)

        if np.sum(y) == 0:
            log.warning("No anomalies in test set!")
            return

        metrics = calculate_all_metrics(y, scores)
        # log metrics
        for metric, value in metrics.items():
            self.log(f"test/{metric}", value, on_epoch=True, prog_bar=True)
        

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers for training.

        :return: A dictionary containing the optimizers and schedulers.
        """
        optimizer1 = self.hparams.optimizer(self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer1)
            return {
                "optimizer": optimizer1,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            }
        return  {"optimizer": optimizer1}

