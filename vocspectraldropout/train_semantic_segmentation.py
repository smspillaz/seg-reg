import argparse
import os
import json
import sys

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

import tqdm

from torch.utils.data import DataLoader

from deeplab_custom_transforms import (
    FixScaleCrop,
    Normalize,
    RandomGaussianBlur,
    RandomHorizontalFlip,
    RandomScaleCrop,
    ToTensor
)

from PIL import Image

from resnet import build_backbone
from decoder import Decoder
from spatial_pyramid_pooling import SpatialPoolingPyramid


class DeepLabModel(nn.Module):
    """DeepLabv3+ Model."""

    def __init__(self, input_channels=3, num_classes=21):
        """Initialize parameters."""
        super().__init__()
        self.feature_detection_layers = build_backbone(input_channels)
        self.spatial_pyramid_pooling = SpatialPoolingPyramid(
            input_channels=2048,
            dilations=(6, 12, 18),
            output_pooling_channels=256
        )
        self.decoder = Decoder(low_level_input_channels=256,
                               low_level_output_channels=48,
                               pyramid_input_channels=256,
                               pyramid_output_channels=256,
                               num_classes=21)

    def forward(self, input):
        x, low_level_features = self.feature_detection_layers(input)
        x = self.spatial_pyramid_pooling(x)
        x = self.decoder(x, low_level_features)

        # Upscale the decoded segmentation into
        # the same size as the input image, where the channels
        # of the decoder output are the segmentation boundaries.
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


def segmentation_cross_entropy_loss(size_average,
                                    ignore_index,
                                    device):
    """Segmentation loss from channels."""
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                    size_average=size_average).to(device)

    def inner(result, target):
        """Divide loss by size of batch."""
        return criterion(result, target.long()) / result.size()[0]

    return inner

class PolynomialLearningRateScheduler(object):
    """Adjust the learning rate on a polynomial curve."""

    def __init__(self, optimizer, lr, epochs, training_set_batches):
        super().__init__()
        self.optimizer = optimizer
        self.base_lr = lr
        self.epochs = epochs
        self.training_set_batches = training_set_batches

    def step(self, epoch, batch_index):
        """Take a step on the learning rate for each of the parameters."""
        total = self.epochs * self.training_set_batches
        current = epoch * self.training_set_batches + batch_index

        # Decrease learning rate on polynomial curve until we reach
        # the end of training.
        lr = self.base_lr * pow(1 - (current / total), 0.9)
        assert lr >= 0

        # First parameter group gets the small learning rate
        self.optimizer.param_groups[0]['lr'] = lr

        # Subsequent parameter groups get 10x much larger learning rate
        for pg in self.optimizer.param_groups[1:]:
            pg['lr'] = lr * 10


def explore_module_children(module):
    for child in module.children():
        if not list(child.children()):
            yield child

        if isinstance(child, nn.Module):
            yield from explore_module_children(child)


def collect_params(module_list):
    for module in module_list:
        for m in explore_module_children(module):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    if p.requires_grad:
                        yield p


def differential_learning_rates(self, module_learning_rates, learning_rate):
    """Get param groups."""
    return [
        {
            "params": list(collect_params(module_list)),
            "lr": learning_rate * multiplier
        }
        for module_list, multiplier in module_learning_rates
    ]


def read_list(path):
    with open(path) as f:
        return [s.strip() for s in f.read().splitlines() if s.strip()]


def check_exists(path):
    """Throws an exception if path does not exist."""
    if not os.path.exists(path):
        raise RuntimeError("Path {} does not exist.".format(path))

    return path


class SpecifiedSegmentationImagesDataset(data.Dataset):
    """Segmentation dataset.

    This contains both the source images and the target segmentations.
    """

    def __init__(self,
                 images_list,
                 source_images_path,
                 target_images_path,
                 transforms):
        super().__init__()
        self.images_list = images_list
        self.source_images_path = source_images_path
        self.target_images_path = target_images_path
        self.transforms = transforms

        self.source_images, self.target_images = zip(*[
            (check_exists("{}.jpg".format(os.path.join(source_images_path, image))),
             check_exists("{}.png".format(os.path.join(target_images_path, image))))
            for image in self.images_list
        ])

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        source_image = Image.open(self.source_images[index]).convert('RGB')

        # Just read the raw data on the target image, each pixel corresponds
        # to one class.
        target_image = Image.open(self.target_images[index])

        # The transforms operate on pairs of images defined as so
        output = self.transforms({
            "image": source_image,
            "label": target_image
        })
        output.update({
            "paths": {
                "image": self.source_images[index],
                "label": self.target_images[index]
            }
        })

        return output

    def with_viewable_transforms(self):
        """Loader with transforms that are still human-viewable."""
        return SpecifiedSegmentationImagesDataset(
            self.images_list,
            self.source_images_path,
            self.target_images_path,
            transforms.Compose([
                t for t in self.transforms.transforms if not isinstance(t, Normalize)
            ])
        )


def load_data(source_images,
              segmentation_images,
              training_set,
              validation_set,
              test_set,
              base_size=513,
              crop_size=513,
              batch_size=8):
    """Create DataLoader objects for each of the three image sets.

    :source_images: are, as the name suggests, the source images.
    :segmentation_images: are the targets for segmentation.
    """
    training_transforms = transforms.Compose([
        RandomHorizontalFlip(),
        RandomScaleCrop(base_size=base_size, crop_size=crop_size),
        RandomGaussianBlur(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
    ])
    validation_transforms = transforms.Compose([
        FixScaleCrop(crop_size=crop_size),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
    ])

    train_dataset = SpecifiedSegmentationImagesDataset(images_list=read_list(training_set),
                                                       source_images_path=source_images,
                                                       target_images_path=segmentation_images,
                                                       transforms=training_transforms)
    validation_dataset = SpecifiedSegmentationImagesDataset(images_list=read_list(validation_set),
                                                            source_images_path=source_images,
                                                            target_images_path=segmentation_images,
                                                            transforms=validation_transforms)
    test_dataset = SpecifiedSegmentationImagesDataset(images_list=read_list(test_set),
                                                      source_images_path=source_images,
                                                      target_images_path=segmentation_images,
                                                      transforms=validation_transforms)

    # Now that we have the datasets, we can create dataloaders for our batch size etc
    return (
        DataLoader(train_dataset,
                   batch_size=batch_size,
                   shuffle=True,
                   pin_memory=True),
        DataLoader(validation_dataset,
                   batch_size=batch_size,
                   shuffle=False,
                   pin_memory=True),
        DataLoader(test_dataset,
                   batch_size=batch_size,
                   shuffle=False,
                   pin_memory=True)
    )


def save_model(path):
    """Save the model to path."""
    def _inner(info):
        """If the path is defined, save the model."""
        if path:
            torch.save(info, path)

    return _inner


def log_statistics(path, append=False):
    """Log the statistics to path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    stream = open(path, "w+" if append else "w")

    def _inner(statistics):
        stream.write(json.dumps(statistics) + "\n")

    return _inner


def calculate_miou(output, target):
    """Calculuate the mean Intersection over Union metric."""
    output = output.cpu().detach().numpy()
    preds = output.argmax(axis=0)
    target = target.cpu().numpy()

    num_classes = output.shape[0]

    # Create a confusion matrix
    mask = (target >= 0) & (target < num_classes)
    label = num_classes * target[mask].astype(int) + preds[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix = count.reshape(num_classes, num_classes)

    # Now compute mean Intersection over Union
    #
    # Diagonal on the matrix is the intersection
    # Sums along the rows and the columns are  the union (with intersection overlap),
    # we must the subtract the intersection in order get the actual union
    union = (
        np.sum(confusion_matrix, axis=1) +
        np.sum(confusion_matrix, axis=0) -
        np.diag(confusion_matrix)
    )

    return np.nanmean(np.diag(confusion_matrix) / union) if union.sum() > 0 else 0


def calculate_many_mious(output_batch, target_batch):
    for output, target in zip(output_batch, target_batch):
        yield calculate_miou(output, target)


def calculate_mean_miou(output_batch, target_batch):
    return np.array(list(calculate_many_mious(output_batch, target_batch))).mean()


def training_loop(model,
                  train_loader,
                  val_loader,
                  criterion,
                  optimizer,
                  scheduler,
                  device,
                  epochs,
                  statistics_callback=None,
                  save_callback=None,
                  start_epoch=0,
                  best_accumulated_miou=0):
    """The main training loop."""
    save_callback = save_callback or (lambda x: None)
    statistics_callback = statistics_callback or (lambda x: None)

    for epoch in tqdm.tqdm(range(start_epoch, epochs + start_epoch), desc="Epoch"):
        model.train()

        progress = tqdm.tqdm(train_loader, desc="Train Batch")
        for batch_index, batch in enumerate(progress):
            source_batch = batch['image'].to(device)
            target_batch = batch['label'].to(device)

            optimizer.zero_grad()

            # Update learning rate
            scheduler.step(epoch, batch_index)

            # Get outputs, evaluate losses etc
            output_batch = model(source_batch)
            loss = criterion(output_batch, target_batch)
            miou = calculate_mean_miou(output_batch, target_batch)

            # Update progress bar
            progress.set_postfix({
                'loss': loss.item(),
                'miou': miou.item(),
            })
            statistics_callback({
                'epoch': epoch,
                'mode': 'train',
                'batch_index': batch_index,
                'statistics': {
                    'loss': loss.item(),
                    'mIoU': miou
                }
            })
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            progress = tqdm.tqdm(val_loader, desc="Validation Batch")

            accumulated_loss = 0
            accumulated_miou = 0

            for batch_index, batch in enumerate(progress):
                source_batch = batch['image'].to(device)
                target_batch = batch['label'].to(device)

                output_batch = model(source_batch)
                loss = criterion(output_batch, target_batch)
                miou = calculate_mean_miou(output_batch, target_batch)
                # Update progress bar
                progress.set_postfix({
                    'loss': loss.item(),
                    'miou': miou
                })
                statistics_callback({
                    'epoch': epoch,
                    'mode': 'validation',
                    'batch_index': batch_index,
                    'statistics': {
                        'loss': loss.item(),
                        'mIoU': miou
                    }
                })

                accumulated_loss += loss.item()
                accumulated_miou += miou

            tqdm.tqdm.write("Epoch {}, Validation loss: {}, Validation mIoU: {}".format(epoch,
                                                                                        accumulated_loss / len(val_loader),
                                                                                        accumulated_miou / len(val_loader)))


            if accumulated_miou > best_accumulated_miou:
                best_accumulated_miou = accumulated_miou
                save_callback({
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_pred": best_accumulated_miou
                })


def call_many(*functions):
    def _inner(*args, **kwargs):
        return [f(*args, **kwargs) for f in functions]

    return _inner


def main():
    """Entry point."""
    parser = argparse.ArgumentParser("Train semantic segmentation model.")
    parser.add_argument("--source-images", type=str, help="Path to source images", required=True)
    parser.add_argument("--segmentation-images", type=str, help="Path to segmented images", required=True)
    parser.add_argument("--training-set", type=str, help="Path to text file containing training set filenames", required=True)
    parser.add_argument("--validation-set", type=str, help="Path to text file containing validation set filenames", required=True)
    parser.add_argument("--test-set", type=str, help="Path to text file containing test set filenames", required=True)
    parser.add_argument("--learning-rate", type=float, default=0.007, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-classes", type=int, default=21, help="Number of segmentation classes")
    parser.add_argument("--save-to", type=str, help="Where to save the model to", required=True)
    parser.add_argument("--log-statistics", type=str, help="Where to log statistics to", default="logs/statistics")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--test-only", action="store_true", help="Only test")
    parser.add_argument("--load-from", type=str, help="Where to load the model from")
    args = parser.parse_args()

    (train_loader,
     val_loader,
     test_loader) = load_data(args.source_images,
                              args.segmentation_images,
                              args.training_set,
                              args.validation_set,
                              args.test_set,
                              batch_size=args.batch_size)
    device = 'cuda' if args.cuda else 'cpu'
    model = DeepLabModel(input_channels=3,
                         num_classes=args.num_classes).to(device)
    print(model)
    criterion = segmentation_cross_entropy_loss(size_average=None,
                                                ignore_index=255,
                                                device=device)
    optimizer = optim.SGD(differential_learning_rates(model, [
                              ((model.feature_detection_layers, ), 1),
                              ((model.spatial_pyramid_pooling, model.decoder), 10)
                          ], args.learning_rate),
                          momentum=0.9,
                          weight_decay=5e-4,
                          nesterov=False)
    start_epoch = 0
    best_accumulated_miou = 0

    if args.load_from:
        saved_info = torch.load(args.load_from)
        start_epoch = saved_info['epoch'],
        best_accumulated_miou = saved_info['best_pred']
        optimizer.load_state_dict(saved_info['optimizer'])
        model.load_state_dict(saved_info['state_dict'])
        print("Loaded model from {}".format(args.load_from))

    scheduler = PolynomialLearningRateScheduler(optimizer,
                                                args.learning_rate,
                                                start_epoch + args.epochs,
                                                len(train_loader))

    if not args.test_only:
        training_loop(model,
                      train_loader,
                      val_loader,
                      criterion,
                      optimizer,
                      scheduler,
                      device,
                      epochs=args.epochs,
                      statistics_callback=log_statistics(args.log_statistics, args.load_from is not None),
                      save_callback=save_model(args.save_to),
                      start_epoch=start_epoch,
                      best_accumulated_miou=best_accumulated_miou)

if __name__ == "__main__":
    main()
