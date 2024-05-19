import time
from typing import Any, Dict, Tuple
import numpy as np

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric

from src import utils
from src.models.components.sar76 import SAR76
from src.utils.record import save_diagno, save_scores
from src.utils.vus import calculate_all_metrics

from src.utils.logging_utils import log


class SAR76Module(LightningModule):
    """SAR76 module for anomaly detection.

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
            # weight for diver. loss
            diver_weight: float = 1e-2,
            # weight for detec. loss
            detec_weight: float = 1e-2,
            # weight for detec. scores
            detec_scores: float = 1.0,
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

        self.net: SAR76 = net

        # loss function
        self.criterion = torch.nn.MSELoss(reduction="none")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_recon_loss = MeanMetric()
        self.train_detec_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_recon_loss = MeanMetric()
        self.val_detec_loss = MeanMetric()

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

        x, y = batch

        # forward pass
        # s shape: [batch_size, n_layers, 2, n_heads, input_size, input_size]
        # q, q_bar shape: [batch_size, n_heads, input_size]
        x_hat, s, q, q_bar = self(x)

        # calculate recon losses
        # shape: [batch_size, input_size]
        l = self.criterion(x_hat, x).mean(dim=1)
        recon_diagno = l.detach()
        recon_losses = l.mean(dim=-1)

        # calculate detection losses
        # shape: [batch_size, input_size]
        l = self.criterion(q, q_bar).mean(dim=tuple(range(1, q.dim()-1)))
        detec_diagno = l.detach()
        detec_losses = l.mean(dim=-1)

        # calculate scores
        # scores = recon_losses + self.hparams.detec_scores * detec_losses
        scores = (recon_losses-self.net.recon_avg)/self.net.recon_std + \
             (detec_losses-self.net.detec_avg)/self.net.detec_std
        
        # alpha = torch.minimum(torch.exp(-scores.detach()), torch.tensor(1.)).unsqueeze(-1)
        alpha = 0.5

        # calculate diagnosis
        diagno = (alpha)*(recon_diagno-self.net.rdiag_avg)/(self.net.rdiag_std) + \
            (1-alpha)*(detec_diagno-self.net.ddiag_avg)/(self.net.ddiag_std)
        # diagno = (1-alpha)*(recon_diagno)/(self.net.recon_avg) + \
        #     (alpha)*(detec_diagno)/(self.net.detec_std)
        # diagno = recon_diagno * detec_diagno
        # diagno = (recon_diagno-self.net.recon_avg) * (detec_diagno-self.net.detec_avg)
        # diagno = recon_diagno * 1/torch.log2(3*torch.argsort(-detec_diagno, dim=-1)+2)
        # diagno = recon_diagno * torch.exp(-torch.argsort(-detec_diagno, dim=-1))
        # diagno = torch.nn.functional.softmax(0.1*detec_diagno) * recon_diagno

        return {
            "recon_losses": recon_losses,
            "recon_scores": recon_losses,
            "recon_diagno": recon_diagno.cpu().numpy(),
            "y": y,
            "scores": scores,
            "detec_losses": detec_losses,
            "detec_scores": detec_losses,
            "detec_diagno": detec_diagno.cpu().numpy(),
            "diagno": diagno.cpu().numpy(),
        }

    def on_train_start(self) -> None:
        if type(self.logger) is WandbLogger:
            log.info(f'Saving code to wandb...')
            import wandb
            wandb.run.log_code(".")
        self.start_time = time.time()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch begins."""
        self.train_loss.reset()
        self.train_recon_loss.reset()
        self.train_detec_loss.reset()
        
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
        recon_loss = output["recon_losses"].mean()
        self.train_recon_loss(recon_loss)
        self.log("train/recon_loss", self.train_recon_loss, on_step=True, on_epoch=True, prog_bar=True)

        detec_loss = output["detec_losses"].mean()
        self.train_detec_loss(detec_loss)
        self.log("train/detec_loss", self.train_detec_loss, on_step=True, on_epoch=True, prog_bar=True)

        loss = recon_loss + self.hparams.detec_weight * detec_loss
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        return {"loss": loss}
    

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when a validation epoch begins."""
        self.val_loss.reset()
        self.val_recon_loss.reset()
        self.val_detec_loss.reset()
        # store scores
        self.val_scores = []
        self.val_recon_scores = []
        self.val_detec_scores = []
        # store diagnosis
        self.val_recon_diagno = []
        self.val_detec_diagno = []
    
    def on_train_end(self) -> None:
        print(f'Elapsed time: {(time.time() - self.start_time)/60:.2f} mins')

    def validation_step(
            self, 
            batch: Tuple[torch.Tensor, torch.Tensor], 
            batch_idx: int
    ) -> Dict[str, Any]:
        """Perform a single validation step.

        :param batch: A tuple containing the input and target tensors.
        :param batch_idx: The index of the batch.
        """

        output = self.model_step(batch)

        # calculate loss
        recon_loss = output["recon_losses"].mean()
        self.val_recon_loss(recon_loss)
        self.log("val/recon_loss", self.val_recon_loss, on_step=False, on_epoch=True, prog_bar=True)

        detec_loss = output["detec_losses"].mean()
        self.val_detec_loss(detec_loss)
        self.log("val/detec_loss", self.val_detec_loss, on_step=False, on_epoch=True, prog_bar=True)

        loss = recon_loss + self.hparams.detec_weight * detec_loss
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # store scores
        self.val_scores.append(output["scores"].detach().cpu().numpy())
        self.val_recon_scores.append(output["recon_scores"].detach().cpu().numpy())
        self.val_detec_scores.append(output["detec_scores"].detach().cpu().numpy())

        # store diagnosis
        self.val_recon_diagno.append(output["recon_diagno"])
        self.val_detec_diagno.append(output["detec_diagno"])
    
    def on_validation_epoch_end(self) -> None:
        scores = np.concatenate(self.val_scores)
        recon_scores = np.concatenate(self.val_recon_scores)
        detec_scores = np.concatenate(self.val_detec_scores)
        recon_diagno = np.concatenate(self.val_recon_diagno)
        detec_diagno = np.concatenate(self.val_detec_diagno)
        
        self.log("val/scores_mean", scores.mean(), on_epoch=True)
        self.log("val/scores_std", scores.std(), on_epoch=True)
        self.log("val/recon_scores_mean", recon_scores.mean(), on_epoch=True)
        self.log("val/recon_scores_std", recon_scores.std(), on_epoch=True)
        self.log("val/detec_scores_mean", detec_scores.mean(), on_epoch=True)
        self.log("val/detec_scores_std", detec_scores.std(), on_epoch=True)

        self.net.recon_avg = torch.tensor(recon_scores.mean())
        self.net.recon_std = torch.tensor(recon_scores.std())
        
        self.net.detec_avg = torch.tensor(detec_scores.mean())
        self.net.detec_std = torch.tensor(detec_scores.std())

        self.net.rdiag_avg = torch.tensor(recon_diagno.mean(0), device=self.device)
        self.net.rdiag_std = torch.tensor(recon_diagno.std(0), device=self.device)

        self.net.ddiag_avg = torch.tensor(detec_diagno.mean(0), device=self.device)
        self.net.ddiag_std = torch.tensor(detec_diagno.std(0), device=self.device)     

    def on_test_start(self) -> None:
        """Lightning hook that is called when testing begins."""
        self.test_ys = []
        # store scores
        self.test_scores = []
        self.test_recon_scores = []
        self.test_detec_scores = []
        # store diagnosis
        self.test_diagno = []
        self.test_recon_diagno = []
        self.test_detec_diagno = []

        # measure complexity and overheads
        self.start_time = time.time()
        total_params = sum(p.numel() for p in self.net.parameters())
        print(f'Num. of params: {total_params}')

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
        self.test_scores.append(output["scores"].detach().cpu().numpy())
        self.test_recon_scores.append(output["recon_scores"].detach().cpu().numpy())
        self.test_detec_scores.append(output["detec_scores"].detach().cpu().numpy())

        self.test_diagno.append(output["diagno"])
        self.test_recon_diagno.append(output["recon_diagno"])
        self.test_detec_diagno.append(output["detec_diagno"])
    
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a testing epoch ends."""
        # get all scores and labels
        y = np.concatenate(self.test_ys)
        if np.sum(y) == 0:
            log.warning("No anomalies in test set!")
            return
        
        # concat scores
        scores = np.concatenate(self.test_scores)
        recon_scores = np.concatenate(self.test_recon_scores)
        detec_scores = np.concatenate(self.test_detec_scores)
        # concat diagnosis
        diagno = np.concatenate(self.test_diagno)
        recon_diagno = np.concatenate(self.test_recon_diagno)
        detec_diagno = np.concatenate(self.test_detec_diagno)
        
        print(f'Elapsed time: {(time.time() - self.start_time)/60:.2f} mins')
        print(f'Inference per sample: {(time.time() - self.start_time)/len(scores)*1e3:.2f}')

        # save diagnosis
        ts = save_diagno(y, diagno, 'SAR76', \
            self.hparams.dataset, 'logs/diagnosis/', \
        )
        save_diagno(y, recon_diagno, 'SAR76-RE', \
            self.hparams.dataset, 'logs/diagnosis/', ts, \
        )
        save_diagno(y, detec_diagno, 'SAR76-DE', \
            self.hparams.dataset, 'logs/diagnosis/', ts, \
        )

        # save scores
        save_scores(y, scores, 0.0, 'SAR76', \
            self.hparams.dataset, 'logs/models/', ts, \
        )
        # calculate metrics
        metrics = calculate_all_metrics(y, scores)
        # log metrics
        for metric, value in metrics.items():
            self.log(f"test/{metric}", value, on_epoch=True, prog_bar=True)

        # save recon scores
        save_scores(y, recon_scores, 0.0, 'SAR76-RE', \
                    self.hparams.dataset, 'logs/models/', ts)
        # calculate metrics
        metrics = calculate_all_metrics(y, recon_scores)
        # log metrics
        for metric, value in metrics.items():
            self.log(f"test/recon_{metric}", value, on_epoch=True, prog_bar=True)

        # save detec. scores
        save_scores(y, detec_scores, 0.0, 'SAR76-DE', \
                    self.hparams.dataset, 'logs/models/', ts)
        # calculate metrics
        metrics = calculate_all_metrics(y, detec_scores)
        # log metrics
        for metric, value in metrics.items():
            self.log(f"test/detec_{metric}", value, on_epoch=True, prog_bar=True)


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

