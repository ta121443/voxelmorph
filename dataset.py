import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms

class CustomMNISTDataset(Dataset):
  def __init__(self, train=True, transform=None):

    self.train = train

    transform = transforms.Compose(
      [transforms.Pad(padding=2),
      transforms.ToTensor(),
      ]
    )

    mnist_train = datasets.MNIST(root='data', train=True, transform=transform)
    self.mnist = [x[0] for x in mnist_train if x[1] == 5]

    if self.train:
      self.mnist = self.mnist[:len(self.mnist) - 1000]
    else:
      self.mnist = self.mnist[len(self.mnist) - 1000:]

    self.data_size = len(self.mnist)
    vol_shape = self.mnist[0].shape[1:]

    self.zero_phi = torch.zeros(*vol_shape, 2)
  
  def __len__(self):
    return len(self.mnist)

  def __get_item__(self):
    idx1 = torch.randint(0, self.data_size, (1,))
    idx2 = torch.randint(0, self.data_size, (1,))
    moving_image = self.mnist[idx1].permute(1, 2, 0)
    fixed_image = self.mnist[idx2].permute(1, 2, 0)

    inputs = [moving_image, fixed_image]
    outputs = [fixed_image, self.zero_phi]
    return (inputs, outputs)

CustomMNISTDataset()