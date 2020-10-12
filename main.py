from generator import Generator
from discriminator import Discriminator
from dataloader import real_data_target, read_latent_space, fake_data_target, load_dataset_chestray, load_dataset_mnist
from datavisualization import plot_loss_evolution, plot_real_fake_comparison
import torch.nn as nn
import torch.optim as optim

#***************************
# CONFIGURACION PARAMETRICA
#***************************

experimento = 1     # 0: MNIST, 1: CovidChestRay
batch_size = 10
epochs = 70
latent_space_size = 100

generator_learning_rate = 0.001
generator_sgd_momentum = 0.9

discriminator_learning_rate = 0.001
discriminator_sgd_momentum = 0.9

# Covid Chest Ray transforms
resize = True
height = 50
width = 50
crop = False
crop_size = 50
grayscale = True

#***************************
# DATASETS
#***************************

dataloader = None

if experimento == 0:
    # DataLoader: MNIST
    dataloader = load_dataset_mnist(root="./datasets",
                                    download=True,
                                    train=True,
                                    batch_size=batch_size)
    cmap = 'binary'

elif experimento == 1:
    # DataLoader: ChestRay COVID
    dataloader = load_dataset_chestray(root="./datasets/chestray/all/train",
                                       batch_size=batch_size,
                                       shuffle=True,
                                       resize=resize,
                                       height=height,
                                       width=width,
                                       crop=crop,
                                       crop_size=crop_size,
                                       grayscale=grayscale)
    cmap = 'gray'

if dataloader is None:
    print("\nExperimento no válido!\n")
    quit()


#***************************
# INICIALIZAR
#***************************

# Se determina el tamaño de la muestra.
# La salida del generador debe corresponder con este tamaño.
# La entrada del discriminador debe corresponder con este tamaño.
sample_size = dataloader.dataset[0][0].nelement()

# Se inicializa el generador de la red GAN
generator = Generator(100, sample_size)
generator_loss = nn.BCELoss()
generator_optimizer = optim.SGD(generator.parameters(),
                                lr=generator_learning_rate,
                                momentum=generator_sgd_momentum)

# Se inicializa el discriminador de la red GAN
discriminator = Discriminator(sample_size, 1)
discriminator_loss = nn.BCELoss()
discriminator_optimizer = optim.SGD(discriminator.parameters(),
                                    lr=discriminator_learning_rate,
                                    momentum=discriminator_sgd_momentum)


print("\n****************************************************************")
print('Comenzando con la ejecucion del experimento')
print("****************************************************************")

# Se determina la resolución de las imagenes generadas

colour_dimension = dataloader.dataset[0][0].shape[0]
height = dataloader.dataset[0][0].shape[1]
width = dataloader.dataset[0][0].shape[2]

print("Alto(H): " + str(height))
print("Ancho(W): " + str(width))
print("Dimension color (C): " + str(colour_dimension))
print("Colour mode: " + cmap)
print("Tamaño entrada (CxHxW): " + str(sample_size))
print("****************************************************************\n")

#***************************
# ENTRENAMIENTO
#***************************

discriminator_loss_storage, generator_loss_storage = [], []

for epoch in range(epochs):

    data_iterator = iter(dataloader)
    batch_number = 0

    # training discriminator
    while batch_number < len(data_iterator):

        # 1. Train the discriminator
        discriminator.zero_grad()
        # 1.1 Train discriminator on real data
        input_real, _ = next(data_iterator)
        discriminator_real_out = discriminator(input_real.reshape(batch_size, sample_size))
        discriminator_real_loss = discriminator_loss(discriminator_real_out, real_data_target(batch_size))
        discriminator_real_loss.backward()
        # 1.2 Train the discriminator on data produced by the generator
        input_fake = read_latent_space(batch_size)
        generator_fake_out = generator(input_fake).detach()
        discriminator_fake_out = discriminator(generator_fake_out)
        discriminator_fake_loss = discriminator_loss(discriminator_fake_out, fake_data_target(batch_size))
        discriminator_fake_loss.backward()
        # 1.3 Optimizing the discriminator weights
        discriminator_optimizer.step()
        discriminator_loss_storage.append(discriminator_fake_loss + discriminator_real_loss)

        # 2. Train the generator
        generator.zero_grad()
        # 2.1 Create fake data
        input_fake = read_latent_space(batch_size)
        generator_fake_out = generator(input_fake)
        # 2.2 Try to fool the discriminator with fake data
        discriminator_out_to_train_generator = discriminator(generator_fake_out)
        discriminator_loss_to_train_generator = generator_loss(discriminator_out_to_train_generator,
                                                               real_data_target(batch_size))
        discriminator_loss_to_train_generator.backward()
        # 2.3 Optimizing the generator weights
        generator_optimizer.step()
        generator_loss_storage.append(discriminator_loss_to_train_generator)

        batch_number += 1

    print('Epoch={}, Discriminator loss={}, Generator loss={}'.format(epoch,
                                                                      discriminator_loss_storage[-1],
                                                                      generator_loss_storage[-1]))



#***************************
# RESULTADOS
#***************************

# Generacion de muestra
noise_for_plot = read_latent_space(batch_size)
generator_output = generator(noise_for_plot)
fake_data_to_plot = generator_output[0].reshape(colour_dimension, height, width)

real_data_to_plot = dataloader.dataset[0][0]  # Primer item del dataset

plot_real_fake_comparison(real_data_to_plot, fake_data_to_plot, cmap)

# Evolucion de loss
plot_loss_evolution(discriminator_loss_storage, generator_loss_storage)
