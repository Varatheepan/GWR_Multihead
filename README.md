# GWR Multihead

## Overview
- Implementation of a Growing When Required (GWR) multihead classifier for class-incremental learning on MNIST.
- Uses a convolutional feature extractor (`MNIST_Net`) and dynamically extends a task-conditioned classifier (`Task_Multi_Net`).
- Requires preprocessed MNIST splits stored in the `class_by_class`, `train.pt`, and `test.pt` files referenced in `GWR_multihead.py`.

## Setup
- Dependencies (PyTorch, torchvision, numpy, networkx)

## Data Preparation
- Prepare the MNIST dataset in the required format:
  ```bash
  python prepare_mnist_data.py
  ```
- This creates:
  - `train.pt` and `test.pt`: Full MNIST training and test datasets
  - `class_by_class/train/class_X.pt`: Per-class training splits (X = 0-9)
  - `class_by_class/test/class_X.pt`: Per-class test splits (X = 0-9)

## Run
- Execute the experiment from this directory:
  ```bash
  python GWR_multihead.py --class-by-class-dir ./class_by_class --train-file ./train.pt --test-file ./test.pt
  ```
- The script trains the incremental GWR backbone per task, updates the classifier heads, and finally reports accuracy metrics.
- To reproduce the clustering sanity check used during development:
  ```bash
  python mnist_test.py --class-by-class-dir ./class_by_class --train-file ./train.pt --test-file ./test.pt
  ```

## Key Files
- `prepare_mnist_data.py`: Downloads and prepares MNIST data in the required format.
- `GWR_multihead.py`: Orchestrates incremental training and evaluation.
- `gwr.py`: Underlying GWR implementations (`gwr3` is used by default).
- `mnist_dataset_class.py`: Dataset utilities for loading the class-incremental MNIST splits.
- `mnist_test.py`: Example evaluation helper for the saved models.
