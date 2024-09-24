from .base_model import FNN_3layers, projection_head

import torch.nn as nn
import torch.optim as optim

from pytorch_lightning import LightningModule
from .metrics import NTXentLoss


class CLModel(nn.Module):
    def __init__(
        self,
        args,
        input_shape_cp,
        input_shape_tx,
        # output_shape,
        # projection_output_shape,
        # fc_units_cp,
        # fc_units_tx,
        # drop_prob,
    ):
        """
        Contrastive Model that taskes two data modalitites and encode them
        by minimizing InfoNCE loss in the embedding space.
        NOTE: xi: cp_feature, xj: tx_feature
        """
        super().__init__()
        self.cp_encoder = FNN_3layers(
            output_shape=args.embedding_shape,
            input_shape=input_shape_cp,
            fc_units=args.fc_units_cp,
            drop_prob=args.drop_prob,
        )
        self.cp_projection_head = projection_head(
            output_shape=args.projection_shape,
            input_shape=args.embedding_shape,
        )
        self.tx_encoder = FNN_3layers(
            output_shape=args.embedding_shape,
            input_shape=input_shape_tx,
            fc_units=args.fc_units_tx,
            drop_prob=args.drop_prob,
        )
        self.tx_projection_head = projection_head(
            output_shape=args.projection_shape,
            input_shape=args.embedding_shape,
        )
        self.cp_encoder.init_parameters()
        self.tx_encoder.init_parameters()
        self.cp_projection_head.init_parameters()
        self.tx_projection_head.init_parameters()

    def forward(self, xis, xjs):
        zis = self.cp_encoder(xis)
        zis = self.cp_projection_head(zis)
        zjs = self.tx_encoder(xjs)
        zjs = self.tx_projection_head(zjs)
        return zis, zjs

    def forward_cp(self, xis):
        zis = self.cp_encoder(xis)
        return zis

    def forward_tx(self, xjs):
        zjs = self.tx_encoder(xjs)
        return zjs


class contrastive_cp_tx_module(LightningModule):
    def __init__(
        self,
        args,
        input_shape_cp,
        input_shape_tx,
        # T_0,
        # lr,
        # weight_decay,
        # batch_size,
        # device,
        # temperature,
        # lambda_weight,
        # fc_units_cp,
        # fc_units_tx,
        # drop_prob,
        # projection_output_shape,
    ):
        """PL module taking care of all train and val steps for
        contrastive learning between cp and tx.
        Here we use AdamW instead of Adam
        """
        super().__init__()
        self.model = CLModel(
            args=args,
            input_shape_cp=input_shape_cp,
            input_shape_tx=input_shape_tx,
        )
        self.loss_function = NTXentLoss(
            batch_size=args.batch_size,
            temperature=args.temperature,
            lambda_weight=args.lambda_weight,
            device=args.device,
        )
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.T_0 = args.T_0

    def forward(self, xi):
        output = self.model.forward_cp(xi)
        return output

    def forward_tx(self, xj):
        output = self.model.forward_tx(xj)
        return output

    def training_step(self, batch, batch_idx):
        xi, xj = batch
        zi, zj = self.model(xi, xj)
        loss = self.loss_function(zi, zj)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xi, xj = batch
        zi, zj = self.model(xi, xj)
        loss = self.loss_function(zi, zj)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.T_0, eta_min=0
        )
        return [optimizer], [scheduler]
