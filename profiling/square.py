import torch
import logging
logging.getLogger().setLevel(logging.DEBUG)
a = torch.compile(torch.square)
a(torch.randn(10))