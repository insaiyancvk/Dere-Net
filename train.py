'''

TODO:

1. import stuff
2. Load augmented data from dataloader
3. Set all functions and constants (loss func, optimizer func, epochs, folders, device etc)
4. Define the model
5. Build a training loop
    1. Move to GPU
    2. Define the training function
    3. Calculate the running loss and store the same in list and later convert to np array and save the np array.
    4. Calculate the accuracy after every batch. Calculate the average of same and store the values in a list 
       and later convert it to np array and save theh np array.
    5. After every epoch, test the model loss and accuracy 
       (by following the same steps as done for training ie, move to list, convert to np array, save the np array) with the test data.

'''

# some image processing tools
import numpy as np
import os, json, warnings
warnings.filterwarnings("ignore")

# the G
import torch
# import torchvision

# again some data loading, preprocessing tools
# from torch.utils.data import DataLoader
# import torchvision.transforms as transformers
# import torchvision.datasets as datasets

# Neural network, optimizer, loss calculator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# a fancy loader
from tqdm import tqdm 
from preprocessor import PreProcessor



# Define constants and set functions

# Load the folders
f = open("FOLDERS.json") # add folders path argument to cli parser
FOLDERS = json.load(f)
f.close()

CLASS_LIST = [
   "Dandere",
   "Deredere",
   "Himdere",
   "Kundere",
   "Tsundere",
   "Yandere",
   "Yangire"
]

BATCH = 16 # add as an argument to CLI

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOSS_FUNC = nn.CrossEntropyLoss()

LR = 0.0005 # add as an argument to CLI

MOMENTUM = 0.9 # add as an argument to CLI

EPOCH = 1



# Load the dataloader
preprocessor = PreProcessor(
   size = 256,
   mean = 0.5,
   std = 0.5,
   BATCH = BATCH
)

train_loader = preprocessor.train_loader(
   train_root = FOLDERS["TRAIN"]
)

test_loader = preprocessor.test_loader(
   test_root = FOLDERS["TEST"]
)

print("\n", DEVICE, "activated.\n")
print("Total number of epochs: ", EPOCH)

# TODO: Check if dirs in FOLDERS exist. Create if they don't. ???
exist = False
for key in FOLDERS:
   if not os.path.isdir(FOLDERS[key]):
      print(key,"directory doesn't exist. Creating.")
      os.mkdir(FOLDERS[key])
      print(key,"directory created.")
   else:
      exist = True
if exist:
   print("\nAll directories exist/created\n")



# The Deep Neural Network

class NeuralNetwork(nn.Module):
    def __init__(self):
        
        super(NeuralNetwork, self).__init__()
        
        self.pool = nn.MaxPool2d(5, 5)
        
        self.conv1 = nn.Conv2d(3, 6, 7)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 3)
        
        self.fc1 = nn.Linear(16, 7)
        #self.fc2 = nn.Linear(500, 50)
        #self.fc3 = nn.Linear(50, 7)

    def forward(self, x):
        
        # Conv layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv layer 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        #print("Tensor shape: ",x.shape)
        
        # Flatten the batch
        x = x.view(x.size(0),-1)
        #print(x.shape)
        
        # Dense layer 1
        x = self.fc1(x)
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

MODEL = NeuralNetwork()
optimizer = optim.SGD(MODEL.parameters(), LR, MOMENTUM)

epoch_loss = 0
epoch_accuracy = 0

def epoch(model, test_loader ,train_loader, loss_func, optim, device):

   train_loss_vals = []
   train_accuracy_vals = []
   test_loss_vals = []
   test_accuracy_vals = []

   train_batch_loss = 0
   train_batch_accuracy = 0

   model = model.to(device)

   print("\n\t\tTraining the batch... \n")

   for image, label in tqdm(train_loader, ncols=100):

      optim.zero_grad()
      
      output = model(image.to(device))
      
      loss = loss_func(
         output,
         label.to(device)
         )
      
      loss.backward()
      optim.step()

      accuracy = np.average(
            np.argmax(
               output.cpu().detach().numpy(), axis=1) == label.cpu().detach().numpy()
         )

      train_loss_vals.append(loss.cpu().item()) # training loss for graph
      train_batch_loss += loss/len(train_loader) # training average loss
      train_accuracy_vals.append(accuracy) # training accuray for graph
      train_batch_accuracy += accuracy/len(train_loader) # training average accuracy
   
   print("\n\t\tValidating the batch... \n")

   for image, label in tqdm(test_loader, ncols=100):

      output = model(image.to(device))

      loss = LOSS_FUNC(
         output,
         label.to(device)
      )

      accuracy = np.average(
         np.argmax(
            output.cpu().detach().numpy(), axis=1) == label.cpu().detach().numpy()
         )

      test_loss_vals.append(loss.cpu().item()) # testing loss for graph
      test_accuracy_vals.append(accuracy) # testing accuracy for graph

   return (

      # TRAIN LOSS
      train_loss_vals, # list
      train_batch_loss, # float

      # TRAIN ACCURACY
      train_accuracy_vals, # list
      train_batch_accuracy, # float

      # TEST LOSS, ACCURACY
      test_loss_vals, # list
      test_accuracy_vals # list
   )

epoch_one = epoch(MODEL, test_loader, train_loader, LOSS_FUNC, optimizer, DEVICE)

print(
   "(Training) batch loss: ", epoch_one[1],"\n"
   "(Validation) batch loss: ", sum(epoch_one[4])/len(epoch_one[4]),"\n"
   "(Training) batch accuracy: ", epoch_one[3], "\n",
   "(Validation) batch accuracy: ", sum(epoch_one[5])/len(epoch_one[5]),"\n"
)