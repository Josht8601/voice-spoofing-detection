import torch
from torch.utils.data import DataLoader
from collections import Counter

from dataset import ASVspoofDataset
from baseline_cnn_model import SimpleCNN
from collections import Counter


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dev dataset
    #dev_dataset = ASVspoofDataset("../data/LA", split="dev")
    #test_systems = ["A16", "A17", "A18", "A19"]
    test_systems = ["A06", "A07", "A08", "A09", "A10"]

    dev_dataset = ASVspoofDataset(
        "../data/LA",
        split="train",
        allowed_systems=test_systems
    )

    labels = [dev_dataset[i][1].item() for i in range(len(dev_dataset))]
    print("FINAL TEST LABEL DISTRIBUTION:", Counter(labels))

    dev_loader = DataLoader(dev_dataset, batch_size=8, shuffle=False)

    # Model
    model = SimpleCNN().to(device)

    # Initialize LazyLinear
    x_init, _ = next(iter(dev_loader))
    x_init = x_init.to(device)
    _ = model(x_init)

    # Load weights
    model.load_state_dict(torch.load("../models/baseline_cnn.pt", map_location=device))
    model.eval()

    correct = 0
    total = 0

    # Diagnostics
    pred_counts = Counter()
    true_counts = Counter()

    tp = fp = tn = fn = 0

    with torch.no_grad():
        for x, y in dev_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            # Count predictions + true labels
            pred_counts.update(preds.cpu().tolist())
            true_counts.update(y.cpu().tolist())

            # Confusion matrix
            for p, t in zip(preds, y):
                if p == 1 and t == 1:
                    tp += 1
                elif p == 1 and t == 0:
                    fp += 1
                elif p == 0 and t == 0:
                    tn += 1
                elif p == 0 and t == 1:
                    fn += 1

    accuracy = correct / total

    print("\n=== RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")

    print("\n=== LABEL DISTRIBUTION ===")
    print("True labels:", true_counts)
    print("Predictions:", pred_counts)

    print("\n=== CONFUSION MATRIX ===")
    print(f"TP (spoof→spoof): {tp}")
    print(f"FP (real→spoof): {fp}")
    print(f"TN (real→real): {tn}")
    print(f"FN (spoof→real): {fn}")


if __name__ == "__main__":
    evaluate()