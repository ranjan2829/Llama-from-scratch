import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd

#1 million characters dataset


lines=open('./input.txt','r').read()
