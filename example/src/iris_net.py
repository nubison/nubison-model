"""IrisNet — vanilla PyTorch model used by ``train_pytorch.ipynb``.

Lives in ``src/`` so that ``pickle.load("src/weights.pkl")`` from
``infer_pytorch.ipynb`` can resolve the class by its qualified name
(``src.iris_net.IrisNet``). Keeping the class inside the train
notebook's ``__main__`` would not survive the round-trip.
"""

import torch.nn as nn


class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))

    def forward(self, x):
        return self.fc(x)
