from pathlib import Path
import struct
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#check cuda availability and set DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {DEVICE}")

#training tuning
batch_size = int(input("batch_size: "))
epochs = int(input("epochs: "))
lr = float(input("learning_rate: "))
momentum = float(input("momentum: "))

#define data directory
data_dir = Path(__file__).resolve().parent / "data"

#define loading images and labels
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
    return DataLoader(dataset, batch_size, shuffle=shuffle)


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
            )

      def forward(self, x):
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = torch.flatten(x, start_dim=1)

            return self.fc(x)
#load model to DEVICE
model = convolution().to(DEVICE)

#select loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr, momentum)

#training loop
total_train = 0

for epoch in range(epochs):
      model.train()
      running_loss = 0.0
      total_train = 0
      for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            scores = model(images)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            total_train += images.size(0)
            running_loss += loss.item() * images.size(0)

      mean_loss = running_loss / total_train
      print(f"epoch {epoch + 1} mean loss={mean_loss:.7f}")

#reset model variables and prime insight variables
model.eval()
test_loss_sum = 0.0
correct = 0
total = 0
class_correct = [0] * 10
class_total = [0] * 10
misclassified = []

#test pass
with torch.no_grad():
      for test_images, test_labels in test_loader:
            test_images = test_images.to(DEVICE)
            test_labels = test_labels.to(DEVICE)

            test_scores = model(test_images)
            batch_loss = criterion(test_scores, test_labels)

            test_loss_sum += batch_loss.item() * test_images.size(0)
            correct += (test_scores.argmax(dim=1) == test_labels).sum().item()
            total += test_labels.size(0)

            probabilities = torch.softmax(test_scores, dim=1)
            confidences, preds = probabilities.max(dim=1)

            for label, pred in zip(test_labels, preds):
                  label_idx = label.item()
                  if label_idx == pred.item():
                        class_correct[label_idx] += 1
                  class_total[label_idx] += 1

            # 3. Collect up to 6 misclassified examples
            wrong_mask = (preds != test_labels)
            if wrong_mask.any() and len(misclassified) < 6:
                  wrong_imgs = test_images[wrong_mask].cpu()
                  wrong_true = test_labels[wrong_mask].cpu()
                  wrong_pred = preds[wrong_mask].cpu()
                  wrong_conf = confidences[wrong_mask].cpu()

                  for img, t_lbl, p_lbl, conf in zip(wrong_imgs, wrong_true, wrong_pred, wrong_conf):
                        if len(misclassified) < 6:
                              misclassified.append((img, t_lbl.item(), p_lbl.item(), conf.item()))

                  for img, t_lbl, p_lbl, conf in zip(wrong_imgs, wrong_true, wrong_pred, wrong_conf):
                        if len(misclassified) < 6:
                              misclassified.append((img, t_lbl.item(), p_lbl.item(), conf.item()))

mean_test_loss = test_loss_sum / total
test_accuracy = correct / total

#intra training insights
print(
      f"epoch: {epoch + 1} "
      f"train loss: {mean_loss:.7f} "
      f"test loss: {mean_test_loss:.7f} "
      f"test accuracy: {test_accuracy:.01%}"
)

#post training insights
print("\n" + "=" * 45)
print(f"test loss:     {mean_test_loss:.7f}")
print(f"test accuracy: {test_accuracy:.01%}")
print("=" * 45)
print("per-class accuracy:")
for digit in range(10):
    digit_acc = class_correct[digit] / class_total[digit] if class_total[digit] > 0 else 0.0
    print(f"  digit '{digit}': {digit_acc:6.2%} ({class_correct[digit]}/{class_total[digit]})")
print("=" * 45)

#visualize
test_dataset = test_loader.dataset
assert isinstance(test_dataset, TensorDataset)

random_index = torch.randint(len(test_dataset), (1,)).item()
image, label = test_dataset[random_index]

model.eval()
with torch.no_grad():
    image_batch = image.unsqueeze(0).to(DEVICE)
    feature_maps = model.conv1(image_batch)

fig, axes = plt.subplots(3, 3, figsize=(8, 8))
axes = axes.flatten()

axes[0].imshow(image.squeeze().numpy(), cmap="gray")
axes[0].set_title(f"input: label {label}")
axes[0].axis("off")

for i in range(8):
    axes[i + 1].imshow(feature_maps[0, i].cpu().numpy(), cmap="viridis")
    axes[i + 1].set_title(f"Filter {i + 1}")
    axes[i + 1].axis("off")

plt.tight_layout()
plt.show()

export = str(input("export model? y for yes ")).strip().lower()

if export == "y":
      model.eval()

      models_dir = Path(__file__).resolve().parent / "models"
      models_dir.mkdir(parents=True, exist_ok=True)

      base_name = "cnn"
      extension = ".pth"
      export_path = models_dir / f"{base_name}{extension}"

      if export_path.exists():
            counter = 1
            while True:
                  candidate_path = models_dir / f"{base_name}{counter:03d}{extension}"
                  if not candidate_path.exists():
                        export_path = candidate_path
                        break
                  counter += 1

      torch.save(model.cpu().state_dict(), export_path)

      print(f"\n[+] Model parameters successfully exported to '{export_path}'")