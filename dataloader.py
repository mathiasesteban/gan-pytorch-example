import torch

from torchvision.datasets import ImageFolder, MNIST

from torchvision.transforms import ToTensor, Compose, CenterCrop, Grayscale, Resize
from torch.autograd import Variable

from datavisualization import plot_image_data

from PIL import Image


def load_dataset_chestray(root, batch_size, shuffle, resize, height, width, crop, crop_size, grayscale):

    # Se define la transformacion compuesta:
    # CROP: Para igualar tamañano de entrada
    # TOTENSOR: Para que el tipo sea el adecuado

    transforms = []

    if grayscale:
        transforms.append(Grayscale(num_output_channels=1))

    if resize:
        transforms.append(Resize(size=[height, width], interpolation=Image.NEAREST))

    if crop:
        transforms.append(CenterCrop(crop_size))

    transforms.append(ToTensor())

    dataset = ImageFolder(root=root, transform=Compose(transforms))

    # plot_image_data(dataset[0][0],'Prueba' , 'gray')

    # <dataset> es una subclase de torch.utils.data.Dataset
    # Implementa los metodos __getitem__(), __len__()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    print("\n*******************************")
    print("Dataset Covid Chest Ray cargado")
    print("Tamaño: " + str(dataset.__len__()))
    print("Batchsize: " + str(batch_size))
    print("Batches: " + str(len(data_loader)))
    print("Clases: " + str(dataset.classes))
    print("Sample shape: " + str(dataset[0][0].shape))
    print("*******************************")


    return data_loader


def load_dataset_mnist(root, train, download, batch_size):
    data_transforms = Compose([ToTensor()])
    mnist_trainset = MNIST(root=root, train=train, download=download, transform=data_transforms)
    dataloader_mnist_train = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

    print("\n*******************************")
    print("Dataset MNIST cargado")
    print("Tamaño: " + str(mnist_trainset.__len__()))
    print("Batchsize: " + str(batch_size))
    print("Batches: " + str(len(dataloader_mnist_train)))
    print("Clases: " + str(mnist_trainset.classes))
    print("Shape: " + str(mnist_trainset[0][0].shape))
    print("*******************************")

    return dataloader_mnist_train


def read_latent_space(size):
    """
    Creates a tensor with random values fro latent space  with shape = size
    :param size: Size of the tensor (batch size).
    :return: Tensor with random values (z) with shape = size
    """
    z = torch.rand(size,100)
    if torch.cuda.is_available(): return z.cuda()
    return z


def real_data_target(size):
    """
    Creates a tensor with the target for real data with shape = size
    :param size: Size of the tensor (batch size).
    :return: Tensor with real label value (ones) with shape = size
    """
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data


def fake_data_target(size):
    """
    Creates a tensor with the target for fake data with shape = size
    :param size: Size of the tensor (batch size).
    :return: Tensor with fake label value (zeros) with shape = size
    """
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data
