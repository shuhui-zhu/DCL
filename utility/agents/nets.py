import torch.nn as nn


def _mlp(input_dim, output_dim, hidden_dim, output_activation=None):
    layers = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    ]
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class SoftmaxNet(nn.Module):
    """MLP that outputs a softmax distribution. Used for proposal/commitment/action actors."""

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = _mlp(input_dim, output_dim, hidden_dim, nn.Softmax(dim=-1))

    def forward(self, x):
        return self.net(x)


class CriticNet(nn.Module):
    """MLP value head returning a scalar."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = _mlp(input_dim, 1, hidden_dim)

    def forward(self, x):
        return self.net(x)
