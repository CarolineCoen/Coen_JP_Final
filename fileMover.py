import pickle
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from City import City
from PIL import Image
import ssl
# to understand the ImageDataGenerator, I used this website: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3

import torch
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn
import torchvision
from torchvision.io import read_image
import torchvision.transforms.functional as transforms
from torch.nn.functional import relu
import tqdm
import torch.optim as optim
from torchvision.datasets import ImageFolder
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.autograd import Variable
import os
import shutil
from PIL import Image
import numpy as np
import tensorflow as tf

def main():
    move()

def move():
    # List all files in the source folder
    files = os.listdir('./LULC-pngs/test/imageTiles/')
    num = 10
    i = 0
    # Iterate through files in the source folder
    for file_name in files:
        if (i >= num):
            break
          # Construct paths for the source and destination images
        source_path = os.path.join('./LULC-pngs/test/imageTiles/', file_name)
        destination_folder = './LULC-pngs/test/imageSubset/'
        destination_path = os.path.join(destination_folder, file_name)
            
            # Move the image file to the destination folder
        shutil.move(source_path, destination_path)
        print(f"Moved '{file_name}' to '{destination_folder}'.")

        file_name = file_name[:-7]
        file_name = file_name+"LC.png"
        source_path = os.path.join('./LULC-pngs/test/maskTiles/', file_name)
        destination_folder = './LULC-pngs/test/maskSubset/'
        destination_path = os.path.join(destination_folder, file_name)
            
            # Move the image file to the destination folder
        shutil.move(source_path, destination_path)
        print(f"Moved '{file_name}' to '{destination_folder}'.")


        i+=1


main()