"""Exported from VisualML -- architecture: input -> conv -> activation -> conv -> activation -> conv -> activation -> pool -> flatten -> linear -> output."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---- config (mirrors your visual graph) ----
IMAGE_SIZE = 28
IN_CHANNELS = 1
NUM_CLASSES = 10
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001


class VisualNet(nn.Module):
    """input -> conv -> activation -> conv -> activation -> conv -> activation -> pool -> flatten -> linear -> output (36170 params)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(3136, 10),
        )

    def forward(self, x):
        return self.net(x)


def demo_data(n=2000):
    X = torch.randn(n, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    s = X.mean(dim=(1, 2, 3))
    y = ((s - s.min()) / (s.max() - s.min() + 1e-6) * NUM_CLASSES).long().clamp(0, NUM_CLASSES - 1)
    return TensorDataset(X, y)


def main():
    torch.manual_seed(0)
    model = VisualNet()
    loader = DataLoader(demo_data(), batch_size=BATCH_SIZE, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()  # loss: cross_entropy
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for X, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(X)
        print(f"epoch {epoch}/{EPOCHS} loss={total / len(loader.dataset):.4f}")
    torch.save(model.state_dict(), "model.pt")
    print("Saved model.pt")


if __name__ == "__main__":
    main()
