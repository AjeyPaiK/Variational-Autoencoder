import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from ModelID import *

batch_size = 4

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
print("No of images in training set = ", len(trainloader))

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
print("No of images in test set = ", len(testloader))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

model = VAE()
criterion = VAE_loss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    running_MSE = 0.0
    running_KLD = 0.0
    print("epoch: ", epoch+1)
    for i, data in tqdm(enumerate(trainloader, 0), total = len(trainloader)/batch_size):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        z, mu, logvar, x_hat = model(inputs)
        loss, MSE, KLD = criterion(x_hat, inputs, mu, logvar)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        running_MSE += MSE.item()
        running_KLD += KLD.item()
        if i % 10 == 9:    # print every 2000 mini-batches
            print('Epoch: %d, Batch: %d, loss: %.3f, Reconstruction loss (MSE): %.2f, KLD: %.2f' % (epoch + 1, i + 1, running_loss / 10, running_MSE/10, running_KLD/10))
            running_loss = 0.0

print('Finished Training')