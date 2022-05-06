from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime
import torch 
from pprint import pprint 

from configs.config import Config
from models.matching.model import JointModel
from models.matching.loss import LossComputer
from datasets.matching import XMatchingDataModule
from models.matching.metric import AverageBatchwiseRecall


class Model(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = JointModel(cfg.MODEL)
        self.loss = LossComputer(cfg)
        self.cfg = cfg 

        # Metric
        self.recalls = cfg.METRIC.RECALL
        self.train_rec = [AverageBatchwiseRecall(recall = recall) for recall in self.recalls]
        self.val_rec = [AverageBatchwiseRecall(recall = recall) for recall in self.recalls]

    def forward(self, v_in, l_in):
        v_out, l_out = self.model(v_in, l_in)
        return v_out, l_out 

    def training_step(self, batch, batch_idx):
        vis, lang = batch
        embeddings = self(vis, lang)
        metrics = [rec(embeddings) for rec in self.train_rec]
        [self.log(f'rec_{self.recalls[i]}/train', metric, on_step=True, on_epoch=False, prog_bar=True, logger=True) for i, metric in enumerate(metrics)]
        loss = self.loss(embeddings)
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vis, lang = batch
        embeddings = self(vis, lang)

        metrics = [rec(embeddings) for rec in self.val_rec]
        [self.log(f'rec_{self.recalls[i]}/val', metric, on_step=True, on_epoch=True, prog_bar=True, logger=True) for i, metric in enumerate(metrics)]
        loss = self.loss(embeddings)
        self.log("loss/val", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR)
        return optimizer


def main():
    cfg_path = 'configs/default.yaml'
    cfg = Config(cfg_path)
    pprint(cfg)
    seed_everything(cfg.SEED)

    # Logger 
    long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), cfg.ID)
    wandb_logger = WandbLogger(name=long_id,project=cfg.PROJECT, log_model="all")
    wandb_logger.experiment.config.update(cfg)

    # Checkpoint 
    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="loss/val",
        mode="min",
        dirpath=f"lightning_logs/{long_id}",
        filename="BEST-VALLOSS-{epoch:02d}",
    )

    # Train
    model = Model(cfg)
    dm = XMatchingDataModule(cfg)
    trainer = Trainer(
        logger=wandb_logger,
        precision=16,
        max_epochs=20,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback],
    )
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()