from torch.utils.data import DataLoader

from nano_embeddings.base.trainer import Trainer

from .config import BgeConfig, BgeTrainerConfig
from .model import BgeM3


class BgeTrainer(Trainer):
    def __init__(self, model: BgeM3, data_loader: DataLoader, trainer_config: BgeTrainerConfig):
        super().__init__()

        self.model = model
        self.model.to(self.device)

        self.data_loader = data_loader
        self.trainer_config = trainer_config


    def train(self):
        for epoch in range(self.trainer_config.epochs):
            self.model.train()
            for batch in self.data_loader:
                #TODO
                pass


    @staticmethod
    def init_trainer_and_model_with_config(config: BgeConfig):
        model = BgeM3(config)
        model.train()
        return BgeTrainer(model)
