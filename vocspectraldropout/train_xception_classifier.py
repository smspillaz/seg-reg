import argparse
import gc

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tqdm import tqdm

from xception import (
    FeatureMapClassifier,
    Xception
)

from other.xception import AlignedXception


IMAGE_NORMALIZATION_WEIGHTS = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


def create_transformations(image_size=299):
    """Create some transformations."""
    # We want to do a few flips, rotations and crops with the training data
    # to regularize a little
    transforms_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size, padding=38),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*IMAGE_NORMALIZATION_WEIGHTS)
    ])

    # For validation we don't want random crops or flips
    transforms_validation = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(*IMAGE_NORMALIZATION_WEIGHTS)
    ])

    return transforms_train, transforms_validation, transforms_validation


def load_datasets_with_transformations(dataset_class,
                                       transforms_train,
                                       transforms_validation,
                                       transforms_test):
    """Load the datasets."""
    data_path = './data'
    return (
        dataset_class(root=data_path, train=True, download=True, transform=transforms_train),
        dataset_class(root=data_path, train=True, download=True, transform=transforms_validation),
        dataset_class(root=data_path, train=False, download=True, transform=transforms_test)
    )



def create_data_loaders(train_dataset, validation_dataset, test_dataset, batch_size=16):
    """Create some loaders for the data that we want to test with."""
    train_indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(train_indices, shuffle=True)

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(validation_dataset,
                                             batch_size=batch_size,
                                             sampler=validation_sampler,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)

    return train_loader, val_loader, test_loader


def train(network, optimizer, train_dataloader, criterion, device):
    """Train the network for one epoch."""
    network.train()
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = network(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        yield loss.item()


def test(network, test_dataloader, device):
    """Train the network for one epoch."""
    network.eval()
    accuracies = []

    for i, data in enumerate(test_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = network(inputs)

        accuracies.append(accuracy_score(torch.argmax(torch.exp(outputs), -1).detach().cpu().numpy().ravel(),
                                         labels.cpu().numpy().ravel()))


    return np.mean(accuracies)


class XceptionClassifier(nn.Module):
    """A classifier that uses exception."""

    def __init__(self, in_channels, out_channels, classes, layers, initialization):
        super().__init__()
        #self.xception = AlignedXception(16, nn.BatchNorm2d, pretrained=False)
        self.xception = Xception(in_channels, layers, out_channels, initialization=initialization)
        self.classifier = FeatureMapClassifier(out_channels, classes)

    def forward(self, x):
        x, _ = self.xception(x)
        return self.classifier(x)



def main():
    """Train classifier on xception, for testing."""
    parser = argparse.ArgumentParser("Train Xception Image Classifier")
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--image-size", type=int, default=299)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=10)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--initialization", action="store_true", default=False)
    args = parser.parse_args()

    np.random.seed(10)

    transforms = create_transformations(args.image_size)
    datasets = load_datasets_with_transformations(torchvision.datasets.CIFAR10,
                                                  *transforms)
    train_dataloader, val_dataloader, test_dataloader = create_data_loaders(*datasets, args.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = XceptionClassifier(3, 2048, 10, args.layers, args.initialization).to(device)
    print(net)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9)

    for epoch in range(args.epochs):
        training_progress = tqdm(train_dataloader, desc='Train epoch {}'.format(epoch))
        for loss in train(net,
                          optimizer,
                          training_progress,
                          criterion,
                          device):
            training_progress.set_postfix(loss=loss)

        gc.collect()
        torch.cuda.empty_cache()

        tqdm.write('Validation set accuracy (epoch {}): {}'.format(epoch,
                                                                   test(net, tqdm(val_dataloader, desc='Validating'), device)))

    tqdm.write('Test set accuracy (epoch {}): {}'.format(epoch,
                                                         test(net, tqdm(test_dataloader, desc='Testing'), device)))

if __name__ == "__main__":
    main()

    
