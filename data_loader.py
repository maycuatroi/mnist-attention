import torch
from torchvision import datasets, transforms

def load_mnist():
    # Define the transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader

# Usage:
# train_loader, test_loader = load_mnist()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Load the MNIST dataset
    train_loader, test_loader = load_mnist()

    # Plot some training images
    plt.figure(figsize=(10, 5))
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.numpy()
        image = images[0].squeeze()  # Remove the extra dimension
        plt.imshow(image, cmap='gray')
        plt.show()
        break
