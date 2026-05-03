import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def set_device():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(device)
    return device

def fc1():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def set_dataloader(bs, seed):
    return DataLoader(
        train_data,
        batch_size=bs,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )

def train(model, optimizer, loader):
    model.train()
    total_loss = 0.0
    count = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count
    # print(avg_loss)

    return avg_loss

###

seed = 1
batch_size = 64
epochs = 20

device = set_device()
set_seed(seed)

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = set_dataloader(batch_size, seed)

model_a = fc1().to(device)
model_b = fc1().to(device)
model_b.load_state_dict(model_a.state_dict())

loss_fn = nn.CrossEntropyLoss()

opt_a = torch.optim.RAdam(model_a.parameters(), lr=1e-3)
opt_b = MyRAdam(model_b.parameters(), lr=1e-3)


results = {
    "torch": [],
    "myradam": [],
}
print(f"step | torch | myradam")
for epoch in range(epochs):

    # train_loader = set_dataloader(batch_size, seed)  
    loss_a = train(model_a, opt_a, train_loader)

    # train_loader = set_dataloader(batch_size, seed)
    loss_b = train(model_b, opt_b, train_loader)
    
    results["torch"].append(loss_a)
    results["myradam"].append(loss_b)

    print(f"{epoch} | {loss_a:.6f} | {loss_b:.6f}")