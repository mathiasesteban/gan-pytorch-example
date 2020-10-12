import matplotlib.pyplot as plt


def plot_loss_evolution(discriminator_loss, generator_loss):
    x = range(len(discriminator_loss)) if len(discriminator_loss) > 0 else range(len(generator_loss))
    if len(discriminator_loss) > 0: plt.plot(x, discriminator_loss, '-b', label='Discriminator loss')
    if len(generator_loss) > 0: plt.plot(x, generator_loss, ':r', label='Generator loss')
    plt.legend()
    plt.show()


def plot_image_data(data, shape):
    pass


def plot_mnist_data(data, label=None):

    data = data.detach().reshape(28, 28)
    plt.imshow(data, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    if label is not None:
        plt.xlabel(label, fontsize='x-large')
    plt.show()


def plot_covid_data(data, label=None):
    # data dimension is (W,H,3) - We are using RGB images.
    #plt.imshow(data.permute(1, 2, 0))
    data = data.detach().reshape(50, 50)

    plt.imshow(data, cmap='gray')
    plt.xticks([])
    plt.yticks([])

    if label is not None:
        plt.xlabel(label, fontsize='x-large')

    plt.show()


def plot_image_data(data, label, cmap):

    colour_dimension = data.shape[0]
    height = data.shape[1]
    width = data.shape[2]

    if colour_dimension > 1:
        data = data.permute(1, 2, 0)
    else:
        data = data.detach().reshape(height, width)

    axis1 = plt.subplot(111)
    plt.imshow(data, cmap=cmap)
    axis1.set_title(label)

    plt.show()


def plot_real_fake_comparison(real_data, fake_data, cmap):

    # Tensor imagen con dimensiones [C,H,W]
    colour_dimension = real_data.shape[0]
    height = real_data.shape[1]
    width = real_data.shape[2]

    # Si es RGB [3,H,W]
    if colour_dimension > 1:
        real_data = real_data.reshape(1, 2, 0)
        fake_data = fake_data.reshape(1, 2, 0)

    # Si no es RGB [1,H,W]
    else:
        real_data = real_data.detach().reshape(height, width)
        fake_data = fake_data.detach().reshape(height, width)


    axis1 = plt.subplot(121)
    plt.imshow(real_data, cmap=cmap)
    axis1.set_title('Real data')

    axis2 = plt.subplot(122)
    plt.imshow(fake_data, cmap=cmap)
    axis2.set_title('Generated data')

    plt.show()
