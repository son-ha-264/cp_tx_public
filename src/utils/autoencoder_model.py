import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.optim as optim


class BimodalAutoencoder(nn.Module):
    def __init__(
        self,
        args,
        cp_dim,
        tx_dim,
        hidden_sizes_encoder_cp,
        hidden_sizes_decoder_cp,
        hidden_sizes_decoder_tx,
    ):
        """PyTorch Module of Bimodal Autoencoder for CP and TX data"""
        super(BimodalAutoencoder, self).__init__()

        self.cp_encoder = self.create_mlp(
            cp_dim, hidden_sizes_encoder_cp, args.drop_prob, args.embedding_shape
        )
        self.cp_decoder = self.create_mlp(
            args.embedding_shape, hidden_sizes_decoder_cp, args.drop_prob, cp_dim
        )
        self.tx_decoder = self.create_mlp(
            args.embedding_shape, hidden_sizes_decoder_tx, args.drop_prob, tx_dim
        )

    def create_mlp(self, input_dim, hidden_sizes, drop_prob, output_dim):
        mlp = torch.nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
        )
        for i in range(len(hidden_sizes) - 1):
            mlp.add_module(
                f"Hidden_layer_{i}", nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            )
            mlp.add_module(f"BatchNorm_{i}", nn.BatchNorm1d(hidden_sizes[i + 1]))
            mlp.add_module(f"ReLU_{i}", nn.ReLU(inplace=True))
            mlp.add_module(f"Dropout_{i}", nn.Dropout(p=drop_prob))

        mlp.add_module("output_layer", nn.Linear(hidden_sizes[-1], output_dim))
        return mlp

    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x_cp):
        latent_cp = self.cp_encoder(x_cp)
        return (self.cp_decoder(latent_cp), self.tx_decoder(latent_cp))

    def get_latent(self, x):
        latent = self.cp_encoder(x)
        return latent


class BimodalAutoencoderModule(LightningModule):
    def __init__(
        self,
        args,
        cp_dim,
        tx_dim,
        hidden_sizes_encoder_cp,
        hidden_sizes_decoder_cp,
        hidden_sizes_decoder_tx,
    ):
        """Lightning Module for training bimodal autoencoder"""
        super(BimodalAutoencoderModule, self).__init__()
        self.model = BimodalAutoencoder(
            args,
            cp_dim,
            tx_dim,
            hidden_sizes_encoder_cp,
            hidden_sizes_decoder_cp,
            hidden_sizes_decoder_tx,
        )
        self.model.init_parameters()
        self.loss_function = nn.MSELoss()
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.T_0 = args.T_0

    def forward(self, x):
        output = self.model.get_latent(x)
        return output

    def training_step(self, batch, batch_idx):
        cp, tx = batch
        cp_recons, tx_recons = self.model(cp)
        loss = (
            self.loss_function(cp_recons, cp) + self.loss_function(tx_recons, tx)
        ) / 2
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cp, tx = batch
        cp_recons, tx_recons = self.model(cp)
        loss = (
            self.loss_function(cp_recons, cp) + self.loss_function(tx_recons, tx)
        ) / 2
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
