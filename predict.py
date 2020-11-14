'''2. Predict
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
'''


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict

import json

import time

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt

from torch.autograd import Variable

import argparse

import functions_train
import functions_predict