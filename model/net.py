#!/usr/bin/env python3
"""Define the neural network model by extending torch.nn.Module class
"""

import numpy as np
import torch.nn as tnn
import torch.nn.functional as F


class Net(tnn.Module):
    """Extend the torch.nn.Module class to define a custom neural network
    """
    def __init__(self, params):
        """
        Initialize the different layers in the neural network
        Args:
            params: (Params) contains hyperparameters
        """
        super(Net, self).__init__()

        self.conv1 = tnn.Conv2d(3, params.num_channels, 3, stride=1, padding=1)
        self.bn1 = tnn.BatchNorm2d(params.num_channels)

        self.fc1 = tnn.Linear(self.num_channels, 10)
        self.dropout_rate = params.dropout_rate

    def forward(self, x):
        """
        Defines the forward propagation through the network
        Args:
            x: (torch.Tensor) contains a batch of images.
        Returns:
            out: (torch.Tensor) output of the network forward pass
        """
        x = self.bn1(self.conv1(x))
        x = F.dropout(F.relu(self.fc1(x)),
                      p=self.dropout_rate,
                      training=self.training)
        return F.log_softmax(x, dim=1)


def loss_fn(outputs, ground_truth):
    """
    Compute the loss given outputs and ground_truth.
    Args:
        outputs: (torch.Tensor) output of network forward pass
        ground_truth: (torch.Tensor) batch of ground truth
    Returns:
        loss: (torch.Tensor) loss for all the inputs in the batch
    """
    criterion = tnn.CrossEntropyLoss()
    loss = criterion(outputs, ground_truth)
    return loss


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) output of the network
        labels: (np.ndarray) ground truth labels
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels)/float(labels.size)


# Maintain all metrics required during training and evaluation.
def get_metrics():
    """Returns a dictionary of all the metrics to be used
    """
    metrics = {'accuracy': accuracy,
              }
    return metrics
