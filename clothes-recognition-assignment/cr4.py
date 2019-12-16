import torch
from matplotlib import pyplot as plt
from torch import nn , optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import helper


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# #image, label = next(iter(trainloader))

# #print the selected clothes image.

# #helper.imshow(image[0,:])
# plt.imshow(image[0,:].numpy().squeeze())
# plt.show()

#model architecture :
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,64)
        self.fc4 = nn.Linear(64,10)

    def forward(self,x):

        #flatten the input tensor image :
        x = x.view(x.shape[0],-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x),dim = 1)

        return x

# Create the model : 
model = Classifier()
# Define the loss
criterion = nn.NLLLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
    
        # Training pass
        # Clear the gradients, do this because gradients are accumulated
        optimizer.zero_grad()
        # Forward pass, then backward pass, then update weights
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        # Take an update step and few the new weights
        optimizer.step() 
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(trainloader)}")


# With the network trained, we can check out it's predictions.

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.resize_(1, 784)

# Calculate the class probabilities (softmax) for img
ps = torch.exp(model(img))

# Plot the image and probabilities
helper.view_classify(img.resize_(1, 28, 28), ps, version='Fashion')