import math
import torch
from torch import nn

upsample_block_num = int(math.log(4, 2))
print(upsample_block_num)

class Generator(nn.Module):
  def __init__(self,scalefactor):
    upsample_block_num = int(math.log(scalefactor, 2))

    super(Generator,self).__init__()

    self.block1=nn.Sequential(
          nn.Conv2d(3,64,kernel_size=9,padding=4),
          nn.PReLU()
        )
    self.block2 = ResidualBlock(64)
    self.block3 = ResidualBlock(64)
    self.block4 = ResidualBlock(64)
    self.block5 = ResidualBlock(64)
    self.block6 = ResidualBlock(64)

    self.block7=nn.Sequential(
          nn.Conv2d(64,64,kernel_size=3,padding=1),
          nn.BatchNorm2d()
        )
    block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
    block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
    self.block8 = nn.Sequential(*block8)
  def forward(self, x):
    block1 = self.block1(x)
    block2 = self.block2(block1)
    block3 = self.block3(block2)
    block4 = self.block4(block3)
    block5 = self.block5(block4)
    block6 = self.block6(block5)
    block7 = self.block7(block6)
    block8 = self.block8(block1 + block7)

    return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))



class ResidualBlock(nn.Module):
  def __init__(self,channels):
    super(ResidualBlock,self).__init__()
    self.conv1=Conv2d(channels,channels,kernel_size=3,padding=1)
    self.bn1=BatchNorm2d(channels)
    self.prelu=PRelu()
    self.conv2=Conv2d(channels,channels,kernel_size=3,padding=1)
    self.bn2=BatchNorm2d(channels)

  def forward(self,x):
    residual=self.conv1(x)
    residual=self.bn1(residual)
    residual=self.prelu(residual)
    residual=self.conv2(residual)  #2nd layer
    residual=self.bn2(residual)

    return x+residual