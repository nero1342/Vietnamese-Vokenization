from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar

import torch 
from pprint import pprint 

from configs.config import Config
from models.matching.model import JointModel
from models.matching.loss import LossComputer
from datasets.matching import XMatchingDataModule


class Model(LightningModule):
    def __init__(self, cfg):
        self.model = JointModel(cfg.MODEL)
        self.loss = LossComputer(cfg)
        self.cfg = cfg 

    def forward(self, v_in, l_in):
        v_out, l_out = self.model(v_in, l_in)
        return v_out, l_out 

    def training_step(self, batch, batch_idx):
        vis, lang = batch
        embeddings = self(vis, lang)
        return self.loss(embeddings)

    def validation_step(self, batch, batch_idx):
        vis, lang = batch
        embeddings = self(vis, lang)
        return self.loss(embeddings)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.TRAIN.LR)
        return optimizer

def main():
    cfg_path = 'configs/default.yaml'
    cfg = Config(cfg_path)
    pprint(cfg)
    model = Model(cfg)
    dm = XMatchingDataModule(cfg)
    trainer = Trainer(
        max_epochs=2,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )
    trainer.fit(model, dm, max_epochs=2,accelerator="auto", devices=1 if torch.cuda.is_available() else None)

main()