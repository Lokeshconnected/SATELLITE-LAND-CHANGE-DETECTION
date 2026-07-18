import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# DEVICE
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# -----------------------------
# DATASET
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("dataset", transform=transform)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

print("Classes:", dataset.classes)
print("Total samples:", len(dataset))

# -----------------------------
# MODEL
# -----------------------------

class ChangeCNN(nn.Module):

    def __init__(self):
        super(ChangeCNN, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Flatten(),
            nn.Linear(64*8*8,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):

        x = self.conv(x)
        x = self.fc(x)

        return x


model = ChangeCNN().to(device)

# -----------------------------
# TRAINING SETUP
# -----------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50

# -----------------------------
# TRAINING LOOP
# -----------------------------

for epoch in range(EPOCHS):

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("Epoch:", epoch+1, "Loss:", total_loss)

# -----------------------------
# SAVE MODEL
# -----------------------------

torch.save(model.state_dict(), "change_model.pth")

print("Model saved as change_model.pth")