######################################################################
import struct
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
######################################################################

#set directory
data_dir = Path(__file__).parent / "data"

#define image and label loader
def load_images(filename):
      raw = np.fromfile(data_dir / filename, dtype=np.uint8)

      magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
      if magic != 2051:
            raise ValueError(f"Unexpected image file magic number: {magic}")

      raw_images = raw[16:].reshape(count, rows, cols)
      images = raw_images.astype(np.float32) / 255.0

      return images

def load_labels(filename):
      raw = np.fromfile(data_dir / filename, dtype=np.uint8)

      magic, count = struct.unpack(">II", raw[:8])
      if magic != 2049:
            raise ValueError(f"Unexpected label file magic number: {magic}")

      labels = raw[8:]
      if len(labels) != count:
            raise ValueError("Label count in header does not match file contents")

      return labels

#load images and labels
train_images = load_images("train-images-idx3-ubyte")
train_labels = load_labels("train-labels-idx1-ubyte")

test_images = load_images("t10k-images-idx3-ubyte")
test_labels = load_labels("t10k-labels-idx1-ubyte")

#define relu(x)
def relu(x):
      return np.maximum(0, x).astype(np.float32)

#define relu derivative
def drelu(x):
      return (x>0).astype(np.float32)

#define convolution
def convolution(image, kernels):
      filter_count, input_channels, kernel_height, kernel_width = kernels.shape
      _, image_height, image_width = image.shape

      output_height = image_height - kernel_height + 1
      output_width = image_width - kernel_width + 1

      output = np.zeros(
            (filter_count, output_height, output_width),
            dtype=np.float32,
      )

      for f in range(filter_count):
            for y in range(output_height):
                  for x in range(output_width):
                        patch = image[
                              :,
                              y:y + kernel_height,
                              x:x + kernel_width,
                        ]
                        output[f, y, x] = np.sum(patch * kernels[f])

      return output

def pool(feature_maps):
      filter_count, height, width = feature_maps.shape

      windows = feature_maps.reshape(
      filter_count, height // 2, 2, width // 2, 2
      ).transpose(0, 1, 3, 2, 4).reshape(
      filter_count, height // 2, width // 2, 4
      )

      switches = windows.argmax(axis=3)
      pooled = np.take_along_axis(
            windows, switches[..., None], axis=3
      )[..., 0]

      return pooled, switches

def pool_backward(pooled_gradient, switches):
      filter_count, pooled_height, pooled_width = pooled_gradient.shape
      window_gradients = np.zeros(
            (filter_count, pooled_height, pooled_width, 4),
            dtype=np.float32,
      )

      np.put_along_axis(
            window_gradients,
            switches[..., None],
            pooled_gradient[..., None],
            axis=3,
      )

      return window_gradients.reshape(
            filter_count, pooled_height, pooled_width, 2, 2
      ).transpose(0, 1, 3, 2, 4).reshape(
            filter_count, pooled_height * 2, pooled_width * 2
      )

def softmax(scores):
      shifted = scores - np.max(scores)
      exp_scores = np.exp(shifted)
      return exp_scores / np.sum(exp_scores)

def train_one(image, label, kernels, weights, bias, learning_rate):
      # Forward pass
      conv_output = convolution(image, kernels)
      activated = relu(conv_output)
      pooled, switches = pool(activated)

      features = pooled.reshape(-1)
      scores = features @ weights + bias
      probabilities = softmax(scores)
      loss = -np.log(probabilities[label] + 1e-9)

      # Backward pass: softmax + cross-entropy gradient
      score_gradient = probabilities.copy()
      score_gradient[label] -= 1

      # Dense layer gradients
      weight_gradient = np.outer(features, score_gradient)
      bias_gradient = score_gradient
      feature_gradient = weights @ score_gradient
      pooled_gradient = feature_gradient.reshape(pooled.shape)

      # Backpropagate through pooling and ReLU
      activated_gradient = pool_backward(pooled_gradient, switches)
      conv_gradient = activated_gradient * drelu(conv_output)

      # Kernel gradients: sum each input patch times its output gradient.
      kernel_gradient = np.zeros_like(kernels)
      _, output_height, output_width = conv_gradient.shape

      for f in range(kernels.shape[0]):
            for c in range(image.shape[0]):
                  for ky in range(kernels.shape[2]):
                        for kx in range(kernels.shape[3]):
                              image_patch = image[
                                    c,
                                    ky:ky + output_height,
                                    kx:kx + output_width,
                              ]
                              kernel_gradient[f, c, ky, kx] = np.sum(
                                    image_patch * conv_gradient[f]
                              )

      # Gradient descent update
      weights -= learning_rate * weight_gradient
      bias -= learning_rate * bias_gradient
      kernels -= learning_rate * kernel_gradient

      return loss, probabilities

def predict_one(image, kernels, weights, bias):
      conv_output = convolution(image, kernels)
      activated = relu(conv_output)
      pooled, _ = pool(activated)
      features = pooled.reshape(-1)
      scores = features @ weights + bias
      return softmax(scores)

# Initialize four trainable filters and the 10-class dense layer.
rng = np.random.default_rng(0)
filter_count = 4
feature_count = filter_count * 13 * 13

kernels = rng.normal(
      0, np.sqrt(2 / 9), size=(filter_count, 1, 3, 3)
).astype(np.float32)

dense_weights = rng.normal(
      0, np.sqrt(2 / feature_count), size=(feature_count, 10)
).astype(np.float32)
dense_bias = np.zeros(10, dtype=np.float32)

# Training loop
sample_count = int(input("sample_size = "))
test_count = int(input("test_size = "))
epochs = int(input("epochs_count = "))
learning_rate = float(input("learning_rate = "))

training_indices = rng.permutation(len(train_images))[:sample_count]

for epoch in range(epochs):
      epoch_losses = []

      # Shuffle the same training subset each epoch.
      epoch_indices = rng.permutation(training_indices)

      for n in epoch_indices:
            image = train_images[n][None, :, :]
            label = int(train_labels[n])

            loss, _ = train_one(
                  image, label, kernels, dense_weights, dense_bias,
                  learning_rate=learning_rate,
            )
            epoch_losses.append(loss)

      correct = 0
      for n in range(test_count):
            image = test_images[n][None, :, :]
            probabilities = predict_one(
                  image, kernels, dense_weights, dense_bias
            )
            prediction = int(np.argmax(probabilities))
            correct += prediction == int(test_labels[n])

      accuracy = correct / test_count
      print(
            f"epoch {epoch + 1}: "
            f"mean loss={np.mean(epoch_losses):.3f}, "
            f"test accuracy={accuracy:.1%}"
      )
