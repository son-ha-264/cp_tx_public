import torch
import torch.nn as nn


class FNN_3layers(nn.Module):
    def __init__(self, output_shape, input_shape, fc_units=512, drop_prob=0.1):
        """
        Simple 3-layers fully-connected neural network for embedding different modalities.
        """
        super(FNN_3layers, self).__init__()
        assert input_shape

        self.classifier = nn.Sequential(
            nn.Linear(input_shape, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_units, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_units, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob),
            nn.Linear(fc_units, output_shape),
        )

        # init
        self.init_parameters()

    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.classifier(x)


class projection_head(nn.Module):
    """
    Simple 1-layer projectio head for training CL.
    """

    def __init__(
        self,
        output_shape,
        input_shape,
    ):

        super(projection_head, self).__init__()
        assert input_shape
        assert output_shape

        self.classifier = nn.Sequential(
            nn.Linear(input_shape, output_shape),
            # nn.ReLU(inplace=True)
        )

        # init
        self.init_parameters()

    def init_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.classifier(x)
