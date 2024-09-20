from cProfile import label

import torch
import torch.nn as nn
import torch.optim as optim
from click.core import batch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from xgboost import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
test_dataset = datasets.MNIST(root='/.data', train=False, transform=transforms, download=True)
train_dataset = datasets.MNIST(root='/.data', train=True, transform=transforms, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss : {running_loss/len(train_loader):.4f}')

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the model on the test images : {100 * correct / total:.2f}%')

torch.save(model.state_dict(), 'NN_model.pth')