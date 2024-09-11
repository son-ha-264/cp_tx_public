import torch
import numpy as np
from torchmetrics.classification import BinaryAUROC, BinaryF1Score
from sklearn.metrics import average_precision_score


class NTXentLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature, lambda_weight, device):
        """Normalized Temperature-scaled Cross Entropy Loss
        Source: https://arxiv.org/pdf/2002.05709v3.pdf
        """
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        # If T up, more uniform prob dist (less 'sharp peaks' in dist), can be helpful e.g. in controlling trade-off between exploration/exploitation
        self.lambda_weight = lambda_weight
        self.device = device

    def softXEnt(self, target, logits):
        """
        Cross Entropy between target matrix (identity matrix) and logits matrix
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim=1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs):
        # Inits
        batch_size = zis.shape[0]
        labels = torch.eye(batch_size, dtype=torch.float)
        labels = labels.to(self.device)

        # Normalize each zi and zj in the batch
        zis = torch.nn.functional.normalize(zis, p=2, dim=1)
        zjs = torch.nn.functional.normalize(zjs, p=2, dim=1)

        # Calculating logits for matmul of zis and zjs, and zjs and zis
        logits_ij = torch.matmul(zis, torch.transpose(zjs, 0, 1)) / self.temperature
        logits_ji = torch.matmul(zjs, torch.transpose(zis, 0, 1)) / self.temperature

        # Calculate Cross Entropy
        loss_ij = self.softXEnt(labels, logits_ij)  # i fix, contrast j
        loss_ji = self.softXEnt(labels, logits_ji)  # j fix, contrast i

        return self.lambda_weight * loss_ij + (1 - self.lambda_weight) * loss_ji
