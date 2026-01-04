import torch, torchtext
print(torch.__version__)          # 2.3.0
from torchtext.vocab import GloVe
g = GloVe(name="6B", dim=100)
print("vector dim:", g["king"].shape)