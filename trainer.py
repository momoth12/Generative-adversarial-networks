import pytorch_lightning as pl
from mnist_datasets import MNISTDataModule
from models import GAN

class TrainerConfig:
    def __init__(self, batch_size=128, max_epochs=5, gpus=1):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.gpus = gpus

    def train_model(self):
        dm = MNISTDataModule(batch_size=self.batch_size)
        model = GAN()
        trainer = pl.Trainer(
            devices=self.gpus, max_epochs=self.max_epochs, accelerator="gpu"
        )
        trainer.fit(model, dm)

if __name__ == "__main__":
    config = TrainerConfig()
    config.train_model()
