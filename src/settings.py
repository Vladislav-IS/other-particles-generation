import torch


N = 2112
P = 2212
pion_masses = {211: 0.13957039, -211: 0.13957039, 111: 0.1349768}
pions = list(pion_masses)
directions = ['x', 'y', 'z']
nucleons = [N, P]
particles = nucleons + pions
label_to_num = {col: i for i, col in enumerate(pions)}
num_to_label = {v: k for k, v in label_to_num.items()}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
