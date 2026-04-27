import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ASVspoofDataset
from baseline_cnn_model import SimpleCNN


def train():
    # Device (CPU for now, GPU later)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset + DataLoader
    # train_dataset = ASVspoofDataset("../data/LA", split="train")
    #train_systems = [f"A{str(i).zfill(2)}" for i in range(1, 16)]
    train_systems = ["A01", "A02", "A03", "A04", "A05"]
    train_dataset = ASVspoofDataset(
        "../data/LA",
        split="train",
        allowed_systems=train_systems
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Model
    model = SimpleCNN().to(device)

    # Initialize LazyLinear
    x, _ = next(iter(train_loader))
    x = x.to(device)
    _ = model(x)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy calculation
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f}")
        print("-" * 40)

    # Save model
    torch.save(model.state_dict(), "../models/baseline_cnn.pt")
    print("Model saved to ../models/baseline_cnn.pt")


if __name__ == "__main__":
    train()