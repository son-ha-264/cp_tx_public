import os
import pathlib
import numpy as np
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.datamodule import cp_tx_dataset
from utils.contrastive_model import contrastive_cp_tx_module


if __name__ == "__main__":

    # PARSE ARGUMENTS
    parser = ArgumentParser()
    parser.add_argument("--version_name", type=str, required=True)
    # General Settings
    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--inference", type=bool, default=True)
    # Model Hyperparameters
    parser.add_argument("--embedding_shape", type=int, default=512)
    parser.add_argument("--projection_shape", type=int, default=256)
    parser.add_argument("--batch_size", type=bool, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--check_val_every_n_epochs", type=int, default=1)
    parser.add_argument("--T_0", type=int, default=101)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lambda_weight", type=float, default=0.5)
    parser.add_argument("--fc_units_cp", type=int, default=1024)
    parser.add_argument("--fc_units_tx", type=int, default=4096)
    parser.add_argument("--drop_prob", type=float, default=0.1)
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--num_devices", type=int, default=2)
    # If you want to run only inference, load model .ckpt file from this path.
    parser.add_argument("--custom_log_path", type=str, default="")

    args = parser.parse_args()

    # INPUT PATHS
    REPO_PATH = pathlib.Path(os.path.realpath(__file__)).parents[1]
    PATH_INDICES_TRAIN = os.path.join(REPO_PATH, "data", "train_set_index.csv")
    PATH_INDICES_VAL = os.path.join(REPO_PATH, "data", "val_set_index.csv")
    PATH_INDICES_TEST = os.path.join(REPO_PATH, "data", "test_set_index.csv")
    PATH_CP_FEATURES = os.path.join(REPO_PATH, "data", "CP_values.npy")
    PATH_TX_FEATURES = os.path.join(REPO_PATH, "data", "TX_values.npy")

    # OUTPUT PATHS
    LOGS_PATH = os.path.join(REPO_PATH, "logs", "CL_pretrain_cp_tx")
    EMBEDDING_OUTPUT_PATH = os.path.join(
        REPO_PATH, "data", "embedding", "CL_pretrain_cp_tx", f"{args.version_name}"
    )
    if not os.path.exists(EMBEDDING_OUTPUT_PATH):
        os.makedirs(EMBEDDING_OUTPUT_PATH)

    # LOAD DATA
    print("Load data...")
    train_data = cp_tx_dataset(
        path_indices=PATH_INDICES_TRAIN,
        path_cp_features=PATH_CP_FEATURES,
        path_tx_features=PATH_TX_FEATURES,
    )
    val_data = cp_tx_dataset(
        path_indices=PATH_INDICES_VAL,
        path_cp_features=PATH_CP_FEATURES,
        path_tx_features=PATH_TX_FEATURES,
    )
    test_data = cp_tx_dataset(
        path_indices=PATH_INDICES_TEST,
        path_cp_features=PATH_CP_FEATURES,
        path_tx_features=PATH_TX_FEATURES,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    train_loader_for_inference = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # INIT FOR MODEL AND TRAINER
    INPUT_SHAPE_CP = len(train_data[3][0])
    INPUT_SHAPE_TX = len(train_data[3][1])
    tb_logger = pl_loggers.TensorBoardLogger(
        LOGS_PATH, version=args.version_name, default_hp_metric=False
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    val_callback = ModelCheckpoint(mode="min", monitor="val_loss", save_top_k=1)

    model = contrastive_cp_tx_module(
        args=args,
        input_shape_cp=INPUT_SHAPE_CP,
        input_shape_tx=INPUT_SHAPE_TX,
    )

    # SANITY CHECK
    print(f"DEVICE: {args.device}")
    print(f"INPUT_SHAPE_CP: {INPUT_SHAPE_CP}")
    print(f"INPUT_SHAPE_TX: {INPUT_SHAPE_TX}")
    print(f"Total number of datapoints: {len(train_data)+len(val_data)+len(test_data)}")
    print("\n")

    # TRAINING STARTS
    trainer = pl.Trainer(
        accelerator=args.device,
        devices=args.num_devices,
        strategy="ddp",
        max_epochs=args.epochs,
        logger=tb_logger,
        gradient_clip_val=5,
        gradient_clip_algorithm="norm",
        check_val_every_n_epoch=args.check_val_every_n_epochs,
        callbacks=[lr_monitor, val_callback],
        precision=args.precision,
    )
    if args.train:
        print("Training...")
        trainer.fit(model, train_loader, val_loader)
        print("Finished Training")
        print("\n")

    # EMBEDDING GENERATION
    if args.inference:
        if args.train:
            PATH_CHECKPOINT = val_callback.best_model_path
            print(PATH_CHECKPOINT)
        else:
            # You can also indicate path to .ckpt file here
            #        PATH_CHECKPOINT = os.path.join(
            #            LOGS_PATH,
            #            "lightning_logs",
            #            args.version_name,
            #            "checkpoints",
            #            "whatever.ckpt"
            #        )
            PATH_CHECKPOINT = (  # Comment this out if PATH_CHECKPOINT is defined above
                args.custom_log_path
            )
            if not os.path.exists(PATH_CHECKPOINT):
                raise Exception("Path to .cpkt file does not exist.")
            print(PATH_CHECKPOINT)
        model = contrastive_cp_tx_module.load_from_checkpoint(
            PATH_CHECKPOINT,
            args=args,
            input_shape_cp=INPUT_SHAPE_CP,
            input_shape_tx=INPUT_SHAPE_TX,
        ).to(args.device)
        # Create float32 embeddings for CP data (train+val+test)
        print("Create embeddings for CP data...")
        final_list_emb_cp = []
        final_list_emb_tx = []
        for loader_name in ["train", "val", "test"]:
            if loader_name == "train":
                loader = train_loader_for_inference
            elif loader_name == "val":
                loader = val_loader
            elif loader_name == "test":
                loader = test_loader
            print(loader_name)
            with torch.no_grad():
                for x_cp, x_tx in loader:
                    x_cp = x_cp.to(args.device)
                    x_tx = x_tx.to(args.device)
                    y_hat_cp = model(x_cp)
                    y_hat_tx = model.forward_tx(x_tx)
                    final_list_emb_cp.append(y_hat_cp)
                    final_list_emb_tx.append(y_hat_tx)
        final_matrix_emb_cp = torch.cat(final_list_emb_cp, 0)
        final_matrix_emb_tx = torch.cat(final_list_emb_tx, 0)
        # Assert that embeddings are float32
        try:
            assert final_matrix_emb_cp.dtype == torch.float32
            assert final_matrix_emb_tx.dtype == torch.float32
        except AssertionError:
            print("Convert to float32...")
            final_matrix_emb_cp = final_matrix_emb_cp.float()
            final_matrix_emb_tx = final_matrix_emb_tx.float()

        print("Sanity check, dimensions of embedding matrix...")
        print(final_matrix_emb_cp.shape)
        print(final_matrix_emb_tx.shape)

        final_matrix_emb_cp = final_matrix_emb_cp.cpu().detach().numpy()
        final_matrix_emb_tx = final_matrix_emb_tx.cpu().detach().numpy()

        np.save(
            os.path.join(
                EMBEDDING_OUTPUT_PATH, f"embedding_{args.version_name}_cp.npy"
            ),
            final_matrix_emb_cp,
        )
        np.save(
            os.path.join(
                EMBEDDING_OUTPUT_PATH, f"embedding_{args.version_name}_tx.npy"
            ),
            final_matrix_emb_tx,
        )

    print("FIN")
