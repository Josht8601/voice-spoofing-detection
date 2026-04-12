from torch.utils.data import DataLoader
from dataset import ASVspoofDataset

dataset = ASVspoofDataset("../data/LA", split="train")

loader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch_x, batch_y in loader:
    print("Batch X:", batch_x.shape)
    print("Batch Y:", batch_y.shape)
    break