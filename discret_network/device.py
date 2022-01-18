import torch

device = torch.device('cuda' if torch.cuda.is_avalable() else 'cpu')
print(device.type())