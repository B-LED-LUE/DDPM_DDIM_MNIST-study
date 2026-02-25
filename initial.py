import torch
import torch.nn as nn
import math
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else 'cpu'
print(device)