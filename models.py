import os

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

import pytorch_lightning as pl


random_seed = 42
torch.manual_seed(random_seed)

BATCH_SIZE=128
AVAIL_GPUS = min(1, torch.cuda.device_count())
NUM_WORKERS=int(os.cpu_count() / 2)

# Detective: fake or no fake -> 1 output [0, 1]
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten the tensor so it can be fed into the FC layers
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)
    
# Generate Fake Data: output like real data [1, 28, 28] and values -1, 1
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.lin1 = nn.Linear(latent_dim, 7*7*64)  # [n, 256, 7, 7]
        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2) # [n, 64, 16, 16]
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2) # [n, 16, 34, 34]
        self.conv = nn.Conv2d(16, 1, kernel_size=7)  # [n, 1, 28, 28]


    def forward(self, x):
        # Pass latent space input into linear layer and reshape
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 7, 7)  #256

        # Upsample (transposed conv) 16x16 (64 feature maps)
        x = self.ct1(x)
        x = F.relu(x)

        # Upsample to 34x34 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)

        # Convolution to 28x28 (1 feature map)
        return self.conv(x)
# TODO: GAN
class GAN(pl.LightningModule):
  def __init__(self, latent_dim=100, lr=0.002):
    super().__init__()
    self.save_hyperparameters()
    self.generator = Generator(latent_dim=self.hparams.latent_dim)
    self.discriminator = Discriminator()
    self.validation_z = torch.randn(6, self.hparams.latent_dim)
    # Disable automatic optimization
    self.automatic_optimization = False
  def forward(self, z):
    return self.generator(z)

  def adversarial_loss(self, y_hat, y):
    return F.binary_cross_entropy(y_hat, y)

  def training_step(self, batch, batch_idx):
    # Access optimizers manually
    opt_g, opt_d = self.optimizers()
    real_imgs, _ = batch
    z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
    z = z.type_as(real_imgs)

    # Generator training
    opt_g.zero_grad()  # Zero out generator gradients
    fake_imgs = self(z)
    y_hat = self.discriminator(fake_imgs)
    y = torch.ones_like(y_hat) # Use ones_like to create a tensor of ones with the same shape and type as y_hat
    g_loss=self.adversarial_loss(y_hat, y)
    self.manual_backward(g_loss) # Manually backpropagate the generator loss
    opt_g.step() # Update generator weights


    # Discriminator training
    opt_d.zero_grad()  # Zero out discriminator gradients
    y_hat_real = self.discriminator(real_imgs)
    y_real = torch.ones(real_imgs.size(0),1)
    y_real=y_real.type_as(y_hat_real)
    real_loss = self.adversarial_loss(y_hat_real, y_real)

    y_hat_fake = self.discriminator(self(z).detach())
    y_fake = torch.zeros(real_imgs.size(0),1)
    y_fake=y_fake.type_as(y_hat_fake)
    fake_loss=self.adversarial_loss(y_hat_fake, y_fake)

    d_loss = (real_loss + fake_loss) / 2
    self.manual_backward(d_loss) # Manually backpropagate the discriminator loss
    opt_d.step() # Update discriminator weights

    log_dict={"g_loss":g_loss, "d_loss":d_loss}
    return {"loss":d_loss.detach(), "progress_bar":log_dict, "log":log_dict}


  def configure_optimizers(self):
    lr=self.hparams.lr
    opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
    opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
    return [opt_g, opt_d]
  def on_epoch_end(self):
    self.plot_imgs()
  def plot_imgs(self):
    z=self.validation_z.type_as(self.generator.lin1.weight)
    sample_imgs = self(z).cpu()
    print("epoch",self.current_epoch)
    fig=plt.figure()
    for i in range(sample_imgs.size(0)):
      plt.subplot(2, 3, i+1)
      plt.tight_layout()
      plt.imshow(sample_imgs.detach().numpy()[i, 0, :, :], cmap="gray_r", interpolation="none")
      plt.title("Generated Data")
      plt.xticks([])
      plt.yticks([])
    plt.show()        