import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path
import os
import matplotlib.pyplot as plt
from typing import Dict, List

class CarDetection(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(CarDetection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_size*30*30, out_features=output_size)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        # print(x.shape)
        return self.classifier(x)

def train(model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        device="cpu"
    ):
    model.train()
    train_loss, accuracy = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.type(torch.float)
        y_pred = model(X).squeeze()
        # print(y_pred.shape)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.round(torch.sigmoid(y_pred))
        accuracy += (y_pred_class == y).sum().item()/len(y_pred)
    train_loss = train_loss/len(dataloader)
    accuracy = accuracy/len(dataloader)
    return train_loss, accuracy

def test(model: nn.Module,
        loss_fn: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device="cpu"
    ):
    model.eval()
    test_loss, accuracy = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y = y.type(torch.float)
            y_pred = model(X).squeeze()
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            
            y_pred_class = torch.round(torch.sigmoid(y_pred))
            accuracy += (y_pred_class == y).sum().item()/len(y_pred)
    test_loss = test_loss/len(dataloader)
    accuracy = accuracy/len(dataloader)
    return test_loss, accuracy

def plot_loss_curves(results: Dict[str, List[float]]):
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    acc = results["train_accuracy"]
    test_acc = results["test_accuracy"]
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, test_acc, label="Test Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def save_model(model, name):
    MODEL_PATH = Path(".")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = MODEL_PATH / name

    print(f"Saving Model to {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

def main():
    transform = transforms.Compose([
        transforms.ColorJitter(),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=Path('./data/train'), transform=transform)
    test_data = datasets.ImageFolder(root=Path('./data/test'), transform=transforms.ToTensor())
    val_data = datasets.ImageFolder(root=Path('./data/val'), transform=transforms.ToTensor())

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=32,
        num_workers=os.cpu_count(),
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=32,
        num_workers=os.cpu_count(),
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_data,
        batch_size=32,
        num_workers=os.cpu_count(),
        shuffle=True
    )

    model = CarDetection(3, 16, 1, 0.3)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

    results = {
    "train_loss": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": []
    }

    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = train(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test(model=model,
                                        dataloader=val_dataloader,
                                        loss_fn=loss_fn)
        
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_acc)
    plot_loss_curves(results)
    test_loss, test_acc = test(model=model,
                               dataloader=test_dataloader,
                               loss_fn=loss_fn)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    save_model(model=model, name='car-detection-model')

if __name__ == "__main__":
    main()