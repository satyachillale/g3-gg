
import torch
from utils.G3 import G3
device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
model = G3(device).to(device)
state_dict = torch.load('./checkpoints/g3_same.pth', map_location=device)
model.load_state_dict(state_dict)
print(model)
