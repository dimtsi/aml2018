import numpy as np
import random;
import math;

import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

import os
os.chdir('/home/dimtsi/Dropbox/UvA/1st Semester/\
Applied Machine Learning/GitHub repo/UVA_AML18/week_3')

#==================== LOADING DATA ======================###
from dataset_utils import load_mnist
train = list(load_mnist(dataset='training', path='datasets'))
train_images = np.array([im[1] for im in train])
train_targets = np.array([im[0] for im in train])

x_train = train_images#[train_targets < 2][:1024]
y_train = train_targets#[train_targets < 2][:1024]
y_train = y_train.reshape((-1, 1))

test = list(load_mnist(dataset='testing', path='datasets'))
test_images = np.array([im[1] for im in test])
test_targets = np.array([im[0] for im in test])

x_test = test_images#[test_targets < 2][:96]
y_test = test_targets#[test_targets < 2][:96]
y_test = y_test.reshape((-1, 1))

x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape((-1, 1, 28, 28))

x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape((-1, 1, 28, 28))

#==================== LOADING DATA TO PYTORCH ======================###
img_train = torch.from_numpy(np.asarray(x_train).astype('float32')).cuda()
label_train = torch.from_numpy(np.asarray(y_train).astype('float32')).cuda()
train_dataset = torch.utils.data.TensorDataset(img_train, label_train)

img_test = torch.from_numpy(np.asarray(x_test).astype('float32')).cuda()
label_test = torch.from_numpy(np.asarray(y_test).astype('float32')).cuda()
test_dataset = torch.utils.data.TensorDataset(img_test, label_test)

#Hyperparameters
learning_rate = 0.01
num_epochs = 5
batch_size = 32

#Torch batch loading
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True);
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True);

#==================== TRAINING ======================###
#Instantiate the Conv Net
from Torch_CNN import *

cnn = CNN().cuda();
#CNN architecture
summary(cnn, (1,28,28))
#loss function and optimizer
criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate);
#Training
history = {'batch': [], 'loss': [], 'accuracy': []}
for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)#.cpu()
            labels = Variable(labels).squeeze(1).long()#.cpu()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(outputs, 1)
            accuracy_train = (labels == argmax.squeeze()).float().mean()*100
            # Show progress
            if (i+1) % 32 == 0:
                log = " ".join([
                  "Epoch : %d/%d" % (epoch+1, num_epochs),
                  "Iter : %d/%d" % (i+1, len(train_dataset)//batch_size),
                  "Loss: %.4f" % loss.item(),
                  "Accuracy: %.4f" % accuracy_train])
                print('\r{}'.format(log), end='')
                history['batch'].append(i)
                history['loss'].append(loss.item())
                history['accuracy'].append(accuracy_train.item())
        print()

ax_1 = plt.subplot()
ax_1.plot(history['loss'], c = 'r')

ax_2 = plt.twinx(ax_1)
ax_2.plot(history['accuracy'])
# ax_2.set_xlim(0,150)
plt.show()
#==================== VALIDATION ======================###
cnn.eval().cuda()
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    labels= labels.squeeze(1)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.float() == labels).sum()
print('Test Accuracy of the model on the 60000 test images: %.4f %%' % (100*correct.item() / total))
