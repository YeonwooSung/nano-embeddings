import torch


class Trainer:
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda_available else 'cpu')

    def train(self):
        raise NotImplementedError
