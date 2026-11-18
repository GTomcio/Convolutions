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

def formatter(image_file):
    images = load_images(image_file).reshape(-1, 1, 28, 28).cpu()
    dataset = TensorDataset(images)
    return DataLoader(dataset)

image_loader = formatter(
    "t10k-images-idx3-ubyte"
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
            self.fc = nn.Sequential(
                 nn.Linear(1352, 762),
                  nn.ReLU(),
                  nn.Linear(762, 420),
                  nn.ReLU(),
                  nn.Linear(420, 200),
                  nn.ReLU(),
                  nn.Linear(200, 100),
                  nn.ReLU(),
                  nn.Linear(100, 10),
                  nn.Softmax(),
            )

      def forward(self, x):
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = torch.flatten(x, start_dim=1)

            return self.fc(x)

model = convolution().to(DEVICE)

counter = 1
with torch.no_grad():
     for images in image_loader:
        images = images.to(DEVICE)
        model_output = model(images)
        print(f"image: {counter}, output: {model_output}")
        counter += 1