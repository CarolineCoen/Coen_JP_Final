import pickle
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from City import City
from PIL import Image
import ssl
# to understand the ImageDataGenerator, I used this website: https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, SubsetRandomSampler
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

ssl._create_default_https_context = ssl._create_unverified_context

# a global variable that all the functions can use
with open('pickled_trainDict', 'rb') as myPickle:
      trainDict = pickle.load(myPickle)

with open('pickled_testDict', 'rb') as myPickle:
      testDict = pickle.load(myPickle)

# big help, another base of this code is: https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
#big help, the base of this code is: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/oxford_iiit_pet/oxford_iiit_pet_dataset_builder.py
# code base is also: https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
# I modified it for my project
def main():
  np.random.seed(0)
  torch.manual_seed(0)
  
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # defining some important variables
  EPOCHS = 20
  BATCH_SIZE = 16 # I was told to start with batch-size of 8
  # learned how to do this conversion from: https://www.datascienceweekly.org/tutorials/convert-list-to-tensorflow-tensor
  #I modified the following code to turn my folders of images into datasets
  #https://stackoverflow.com/questions/76679892/folder-structure-for-tensorflow-image-segmentation

  train_dataset = CustomImageDataset(masks_dir='./LULC-pngs/train/maskTiles/', img_dir='./LULC-pngs/train/imageTiles/') #iterable = generator(264974))
  validation_dataset = CustomImageDataset(masks_dir='./LULC-pngs/test/maskTiles/', img_dir='./LULC-pngs/test/imageTiles/') #iterable = generator(89167)

# train_loader returns batches of training data. See how train_loader is used in t
# he Trainer class later
  train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last = True)
  validation_loader = DataLoader(dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last = True)
  dataiter = iter(train_loader)
  learning_rate = 1/50

  model = Model()
  model = model.to(device)
  opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  loss_function = nn.CrossEntropyLoss()

  trainer = Trainer(net=model, optim=opt, loss_function=loss_function, train_loader=dataiter)
  
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  writer = SummaryWriter('runs/satellite_trainer_{}'.format(timestamp))
  epoch_number = 0

  avg_losses = []
  best_vloss = 100000

  for epoch in range(EPOCHS):
      model.train(True)
      avg_loss = trainer.train_epoch()
      avg_losses.append(avg_loss)
      print("epoch [%d]: loss %.3f" % (epoch+1, avg_losses[-1]))

      running_validation_loss = 0.0
      # Set the model to evaluation mode, disabling dropout and using population
      # statistics for batch normalization.
      model.eval()

     # Disable gradient computation and reduce memory consumption.
      with torch.no_grad():
          for i, val_data in enumerate(validation_loader):
              valInputs = X = val_data[0]
              valLabels = val_data[1]
              valLabels = torch.squeeze(valLabels, dim=1)
              valOutputs = model(valInputs)
              valLoss = loss_function(valOutputs, valLabels)
              running_validation_loss += valLoss

      avg_vloss = running_validation_loss / (i + 1)
      print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
      writer.add_scalars('Training vs. Validation Loss',
                      { 'Training' : avg_loss, 'Validation' : avg_vloss },
                      epoch_number + 1)
      writer.flush()

    # Track best performance, and save the model's state
      if avg_vloss < best_vloss:
          best_vloss = avg_vloss
          model_path = 'model_{}_{}'.format(timestamp, epoch_number)
          torch.save(model.state_dict(), model_path)

      epoch_number += 1

# [58, 189, 234, 179, 188, 106, 81, 74, 0, 134] these are the values of the grayscale integer pixels in my grayscale masks
  
  print("The model has been trained")
  #https://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image to learn how to save an array as a png
  #Aba1Array = model.predict('./LULC-pngs/test/imageTiles/Aba1_0_0_RGB.png', batch_size=1)
  #Aba1Prediction = Image.fromarray(Aba1Array)
  #Aba1Prediction.save("Aba1_0_0_Predicted.png")

  print("I made it to the end of the code!")

def generator(numItems):
    for i in range(numItems):
        yield i
        

class Model(nn.Module):
   # trying to make a Unet encoder/decoder model based off of this: https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3
    # this is also adapted off of the COS 324 code
    def __init__(self):
        super(Model, self).__init__()

        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) 
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) 
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, 10, kernel_size=1)
        self.final = nn.Softmax(dim=0)

    def forward(self, x):
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer

        # fixed an error with https://discuss.pytorch.org/t/softmax-not-implemented-for-long/102961
        outConv = self.outconv(xd42)
        outConv = outConv.float()
        out = self.final(outConv)
        out = out.float()
        # get the argmax
        out = torch.argmax(out, dim=1)
        out = out.float()

        return out


class Trainer():
    def __init__(self, net=None, optim=None, loss_function=None, train_loader=None):
        self.net = net
        self.optim = optim
        self.loss_function = loss_function
        self.train_loader = train_loader

    def train_epoch(self):
        epoch_loss = 0.0
        epoch_steps = 0
        for data in tqdm.tqdm(self.train_loader):
            # Moving this batch to GPU
            # Note that X has shape (batch_size, number of channels, height, width)
            # which should be equal to (8,3,128,128) since our default batch_size = 8 and
            # the image has 3 channels
            #X, y = next(self.train_loader)
            X = data[0]

            #X = transforms.to_tensor(X)
            y = data[1]

            #y = transforms.to_tensor(y)

            y = torch.squeeze(y, dim=1)

            self.optim.zero_grad()
            output = self.net(X)
            loss = self.loss_function(output, y)
            loss.requires_grad = True
            # solved this issue with: https://stackoverflow.com/questions/61808965/pytorch-runtimeerror-element-0-of-tensors-does-not-require-grad-and-does-not-ha
            loss.backward()
            self.optim.step()

            epoch_loss += loss.item()
            epoch_steps += 1   
        return epoch_loss/epoch_steps


class CustomImageDataset(Dataset):
    def __init__(self, masks_dir, img_dir, transform=None, target_transform=None): #removed iterable
        self.img_labels = masks_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        i=0
        #self.iterable = iterable

    #def __iter__(self):
        #return self.iterable

    def __len__(self):
        i=0
        for file in os.listdir(self.img_dir):
            i+=1
        return i

    def __getitem__(self, idx): # idx might be necessary
        if (self.img_labels[:16] == './LULC-pngs/test'):
            name = testDict[idx]
        else:
            name = trainDict[idx]
        img_path = self.img_dir+name+'RGB.png'
        image = read_image(img_path)
        image = image.float()
        label_path = self.img_labels+name+'LC.png'
        label = read_image(label_path)
        label = label.int()
        label = label.numpy()
        label = correctVals(label)
        label = torch.from_numpy(label)
        label = label.float()
        return image, label 

def correctVals(arr):
    for j in range(128):
      for k in range(128):
          val = arr[0][j][k]
          if (val == 58):
              arr[0][j][k] = 1
          elif (val == 189):
              arr[0][j][k] = 2
          elif (val == 234):
              arr[0][j][k] = 3
          elif (val == 179):
              arr[0][j][k] = 6
          elif (val == 188):
              arr[0][j][k] = 4
          elif (val == 106):
              arr[0][j][k] = 9
          elif (val == 81):
              arr[0][j][k] = 8
          elif (val == 74):
              arr[0][j][k] = 5
          elif (val == 0):
              arr[0][j][k] = 7 #moss and lichen?
          elif (val == 134):
              arr[0][j][k] = 10 #mangroves?
          else:
              print("Bad duck! %d" %(val))
    return arr

main()