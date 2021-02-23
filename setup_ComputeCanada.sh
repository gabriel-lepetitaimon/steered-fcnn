#!/bin/bash
pip install --no-index --upgrade pip
pip install numpy torch torchvision mlflow  opencv_python PyYAML scikit-learn orion h5py --no-index
pip uninstall mlflow orion
pip install pytorch-lightning albumentations pysftp orion mlflow
