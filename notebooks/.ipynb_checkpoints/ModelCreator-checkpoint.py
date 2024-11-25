import torch
import os
from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor
from pathlib import Path
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch import nn
import torch
import numpy as np
import tomllib
from mads_hackathon.metrics import caluclate_cfm
from imblearn.over_sampling import RandomOverSampler
from mads_hackathon import datasets, metrics
# Load the saved model
loaded_model = torch.load("../models/CNNTrainedModel.pth")
matrixshape = (8, 24)
from mads_hackathon.models import CNNConfig as Config
config = Config(
    matrixshape = (8,24),
    batchsize = 64,
    input_channels = 1,
    hidden = 16,
    kernel_size = 3,
    maxpool = 2,
    num_layers = 1,
    num_classes = 5,
)
# Set the model to evaluation mode
model = loaded_model.eval()
print("Model loaded successfully!")
datadir = Path('../hackathon-data/')
validfile = (datadir / "heart_big_valid.parq").resolve()
testdataset = datasets.HeartDataset2D(validfile, target="target", shape=matrixshape)
teststreamer = BaseDatastreamer(testdataset, preprocessor = BasePreprocessor(), batchsize=config.batchsize)
cfm = caluclate_cfm(model, teststreamer)
#for i, tp in enumerate(np.diag(cfm)):
  #  mlflow.log_metric(f"TP_{i}", tp)
plot = sns.heatmap(cfm, annot=cfm, fmt=".3f")
plot.set(xlabel="Predicted", ylabel="Target")