from torch.utils.data import DataLoader
from dataset import ASVspoofDataset
from baseline_cnn_model import SimpleCNN

dataset = ASVspoofDataset("../data/LA", split="train")

loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SimpleCNN()

x, y = next(iter(loader))

output = model(x)

print(output.shape)