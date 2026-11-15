from pathlib import Path
import struct
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {DEVICE}")

data_dir = Path(__file__).resolve().parent / "data"

def load_images(filename):
    with (data_dir / filename).open("rb") as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(
                f"Invalid magic number {magic} in MNIST image file: {filename}"
            )

        image_bytes = np.frombuffer(file.read(), dtype=np.uint8).copy()

    images = torch.from_numpy(image_bytes).reshape(num, rows * cols)
    return images.to(device=DEVICE, dtype=torch.float32).div_(255.0)

def load_labels(filename):
      raw = np.fromfile(data_dir / filename, dtype=np.uint8)

      magic, count = struct.unpack(">II", raw[:8])
      if magic != 2049:
            raise ValueError(f"Unexpected label file magic number: {magic}")

      labels = raw[8:]
      if len(labels) != count:
            raise ValueError("Label count in header does not match file contents")

      return labels

def formatter(image_file, label_file, shuffle):
    images = load_images(image_file).reshape(-1, 1, 28, 28).cpu()
    labels = torch.tensor(load_labels(label_file), dtype=torch.long)

    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=64, shuffle=shuffle)


train_loader = formatter(
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    shuffle=True,
)

test_loader = formatter(
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
    shuffle=False,
)

class convolution(nn.Module):
      def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
      )
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.fc = nn.Linear(8 * 13 * 13 * 10, 1352)

      def forward(self, x):
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = torch.flatten(x, start_dim=1)

            return self.fc(x)

model = convolution().to(DEVICE)

images, labels = next(iter(train_loader))
scores = model(images.to(DEVICE))

print("input batch:", images.shape)
print("score batch:", scores.shape)

