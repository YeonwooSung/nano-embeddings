import torch

# custom modules
from model import BgeM3
from config import BgeConfig, get_config_for_tiny


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    config: BgeConfig = get_config_for_tiny()
    model = BgeM3(config)
    model.to(DEVICE)

    # set model to train mode
    model.train()

    #TODO load dataset and dataloader


if __name__ == '__main__':
    train()
