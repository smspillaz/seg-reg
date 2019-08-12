import argparse
import fnmatch
import os
import itertools
import json
import sys

import numpy as np
import seaborn as sns

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms

import tqdm

from collections import defaultdict
from contextlib import contextmanager

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

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
from patch_dropout import DropoutActivationPatch

from utils.visualization import overlay_segmentation

sns.mpl.use('Agg')


class DeepLabModel(nn.Module):
    """DeepLabv3+ Model."""

    def __init__(self,
                 input_channels=3,
                 num_classes=21,
                 drop_layer_cls=None,
                 feature_detection_dropout_rate=0.0,
                 pyramid_dropout_rate=0.0,
                 decoder_dropout_rate=0.0,
                 decoder_use_channel_attention=False):
        """Initialize parameters."""
        super().__init__()
        self.feature_detection_layers = build_backbone(input_channels,
                                                       drop_layer=drop_layer_cls(feature_detection_dropout_rate) if drop_layer_cls else None)
        self.spatial_pyramid_pooling = SpatialPoolingPyramid(
            input_channels=2048,
            dilations=(6, 12, 18),
            output_pooling_channels=256,
            drop_layer=drop_layer_cls(pyramid_dropout_rate) if drop_layer_cls else None
        )
        self.decoder = Decoder(low_level_input_channels=256,
                               low_level_output_channels=48,
                               pyramid_input_channels=256,
                               pyramid_output_channels=256,
                               num_classes=21,
                               drop_layer=drop_layer_cls(decoder_dropout_rate) if drop_layer_cls else None,
                               use_channel_attention=decoder_use_channel_attention)

    def forward(self, input):
        x, low_level_features = self.feature_detection_layers(input)
        x = self.spatial_pyramid_pooling(x)
        x = self.decoder(x, low_level_features)

        # Upscale the decoded segmentation into
        # the same size as the input image, where the channels
        # of the decoder output are the segmentation boundaries.
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x


    def optimizer_params(self):
        """Get params and learning rates for optimizer."""
        return [
            ((self.feature_detection_layers, ), 1),
            ((self.spatial_pyramid_pooling, self.decoder, ), 10)
        ]


class DeepLabModelZooModel(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.feature_detection_layers = backbone
        self.decoder = classifier

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.feature_detection_layers(x)['out']
        x = self.decoder(x)
        return F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

    def optimizer_params(self):
        """Get params and learning rates for optimizer."""
        return [
            ((self.feature_detection_layers, ), 1),
            ((self.decoder, ), 10)
        ]


def segmentation_cross_entropy_loss(size_average,
                                    ignore_index):
    """Segmentation loss from channels."""
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                    size_average=size_average)

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
        return sorted([s.strip() for s in f.read().splitlines() if s.strip()])


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
            },
            "label_palette": target_image.getpalette()
        })

        return output

    def with_deterministic_transforms(self):
        """Loader with transforms that are always the same."""
        return SpecifiedSegmentationImagesDataset(
            self.images_list,
            self.source_images_path,
            self.target_images_path,
            transforms.Compose([
                t for t in self.transforms.transforms if not any([
                    isinstance(t, cls)
                    for cls in (RandomScaleCrop, RandomGaussianBlur, RandomHorizontalFlip)
                ])
            ])
        )


    def with_viewable_transforms(self):
        """Loader with transforms that are still human-viewable."""
        return SpecifiedSegmentationImagesDataset(
            self.images_list,
            self.source_images_path,
            self.target_images_path,
            transforms.Compose([
                t for t in self.transforms.transforms if not any([
                    isinstance(t, cls)
                    for cls in (Normalize, RandomScaleCrop, RandomGaussianBlur, RandomHorizontalFlip)
                ])
            ])
        )


def bound_list_by_proportion(source_list, proportion, alignment):
    """Bound list by proportion, to the next alignment."""
    bound = int(len(source_list) * proportion)
    bound += bound % alignment
    bound = min(bound, len(source_list))

    return source_list[:bound]


def list_proportion(source_list, proportion, classes_list, batch_size):
    """Get a proportion of the list.
    
    If we don't have a list of classes, we do this the naive way
    and just bound the list directly.

    Otherwise if we have classes we can be a little smarter. Figure out
    where the class list overlaps with the list itself using set
    intersections, then limit the proportion of data within each
    class list.
    """
    if not classes_list:
        return bound_list_by_proportion(source_list, proportion, batch_size)

    source_list_set = set(source_list)
    classes_list_sets = {
        k: bound_list_by_proportion(list(set(v) & source_list_set), proportion, 1)
        for k, v in classes_list.items()
    }

    return sorted(list(itertools.chain.from_iterable(classes_list_sets.values())))


def load_data(source_images,
              segmentation_images,
              training_set,
              validation_set,
              test_set,
              base_size=513,
              crop_size=513,
              batch_size=8,
              limit_train_data=1.0,
              limit_validation_data=1.0,
              classes_list=None):
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

    train_dataset = SpecifiedSegmentationImagesDataset(images_list=list_proportion(read_list(training_set),
                                                                                   limit_train_data,
                                                                                   classes_list,
                                                                                   batch_size),
                                                       source_images_path=source_images,
                                                       target_images_path=segmentation_images,
                                                       transforms=training_transforms)
    validation_dataset = SpecifiedSegmentationImagesDataset(images_list=list_proportion(read_list(validation_set),
                                                                                        limit_validation_data,
                                                                                        classes_list,
                                                                                        batch_size),
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
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(info, path)

    return _inner


def log_statistics(path):
    """Log the statistics to path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    stream = open(path, "a+")

    def _inner(statistics):
        stream.write(json.dumps(statistics) + "\n")
        stream.flush()

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


def segmentation_to_image(segmented_image, num_classes=21, palette=None):
    """Create an image with the segmentation result."""
    img = Image.fromarray(segmented_image.astype('uint8', order='C'), mode='P')

    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    img.putpalette(colors)
    return img


def visualize_segmentation(segmented_image, num_classes=21, palette=None):
    """Visualize segmentation."""
    img = segmentation_to_image(segmented_image, num_classes=21, palette=palette)
    fig = hide_axis(sns.mpl.pyplot.imshow(img).get_figure())
    return fig


def save_segmentation(segmented_image, path, num_classes=21, palette=None):
    """Save a segmentation somewhere."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    visualize_segmentation(segmented_image, palette=palette).savefig(path,
                                                                     bbox_inches="tight",
                                                                     pad_inches=0)
    sns.mpl.pyplot.clf()


@contextmanager
def evaluation(model):
    """Put the model into evaluation mode and disable backprop."""
    train = model.training

    with torch.no_grad():
        try:
            model.eval()
            yield
        finally:
            if train:
                model.train()


def segment_image(model, input):
    """Segment an image and return the predictions and output."""
    with evaluation(model):
        output = model(input.unsqueeze(0))
        pred = torch.argmax(output.detach()[0], dim=0).cpu().byte().numpy()
        return pred, output[0]


def segment_and_save(model, input, target, image_path, log_path, epoch, palette=None):
    """Segment a single image image and save it."""
    pred, output = segment_image(model, input)
    miou = calculate_miou(output.detach(), target)
    save_segmentation(pred, image_path, palette=palette)

    with open(log_path, "a+") as miou_log:
        miou_log.write("{} {}\n".format(epoch, miou))

def splice_into_path(path, splice):
    """Splice something into path, before the extension."""
    dirname = os.path.dirname(path)
    filename, ext = os.path.splitext(os.path.basename(path))

    return os.path.join(dirname, "{}.{}{}".format(filename, splice, ext))


def hide_axis(figure):
    """Hide axis on figure."""
    figure.axes[0].get_xaxis().set_visible(False)
    figure.axes[0].get_yaxis().set_visible(False)
    return figure


def tensor_to_image(tensor):
    """Convert torch tensor to image."""
    return tensor.cpu().numpy().transpose(2, 1, 0) / 255.0


def save_input_image(input, path):
    """Save the input image tensor somewhere."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = tensor_to_image(input)
    hide_axis(sns.mpl.pyplot.imshow(img).get_figure()).savefig(path,
                                                               bbox_inches="tight",
                                                               pad_inches=0)
    sns.mpl.pyplot.clf()


def save_segmentations_for_image(model, viewable_input, input, label, path, palette=None):
    """Do segmentation for a few images and save the result to a PNG."""
    # First, save the input in path too
    save_input_image(viewable_input, splice_into_path(path, "input"))
    save_segmentation(label.cpu().numpy(), splice_into_path(path, "label"), palette=palette)

    def _inner(statistics):
        """Use statistics to save the file."""
        if statistics["mode"] != "train" or statistics["batch_index"] != 0:
            return

        epoch_path = splice_into_path(path, "epoch.{:02d}".format(statistics["epoch"]))
        log_path = "{}.log.txt".format(path)
        segment_and_save(model, input, label, epoch_path, log_path, statistics["epoch"], palette=palette)

    return _inner


def save_model_and_optimizer(path):
    def _inner(model, optimizer, val_loader, mious, epoch):
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_pred": mious.mean()
            }, path)

    return _inner


def save_model_on_better_miou(path, initial_best_miou):
    """Save the model if we have a better miou."""
    best_miou = initial_best_miou
    save_func = save_model_and_optimizer(path)

    def _inner(model, optimizer, val_loader, mious, epoch):
        nonlocal best_miou
        miou = mious.mean()
        if miou > best_miou:
            best_miou = miou
            save_func(model, optimizer, val_loader, mious, epoch)

    return _inner


def save_interesting_images(path, device):
    """Save some interesting images from the validation process on each epoch.

    'Interesting' is defined as the three images with the best segmentation,
    the three images with the worst segmentation and the three images in the
    middle.
    """
    def save_pairs(model, pairs, epoch, tag):
        """Helper to save pairs of images."""
        for i, pair in enumerate(pairs):
            network_input, viewable = pair
            epoch_path = splice_into_path(path, ".".join([tag, str(i), "epoch{:02d}".format(epoch)]))
            save_input_image(viewable["image"], splice_into_path(epoch_path, "input"))
            save_segmentation(viewable["label"].cpu().numpy(),
                              splice_into_path(epoch_path, "label"),
                              palette=viewable["label_palette"])
            segment_and_save(model,
                             network_input["image"].to(device),
                             network_input["label"].cpu(),
                             splice_into_path(epoch_path, "segmentation"),
                             splice_into_path(path, ".".join([tag, str(i)])) + ".log.txt",
                             epoch,
                             palette=network_input["label_palette"])

    def _inner(model, optimizer, val_loader, mious, epoch):
        ordered_mious = mious.argsort()
        viewable_val_loader = val_loader.dataset.with_viewable_transforms()

        length = len(ordered_mious)
        middle = (length - 1) // 2
        worst_three = [(val_loader.dataset[i], viewable_val_loader[i]) for i in ordered_mious[:3]]
        best_three = [(val_loader.dataset[i], viewable_val_loader[i]) for i in ordered_mious[-3:]]
        middle_three = [(val_loader.dataset[i], viewable_val_loader[i]) for i in ordered_mious[middle - 1:middle + 1]]

        # Now that we have the worst, best and middle images,
        # save them
        save_pairs(model, worst_three, epoch, "worst")
        save_pairs(model, best_three, epoch, "best")
        save_pairs(model, middle_three, epoch, "middle")

    return _inner


def validate(model,
             val_loader,
             criterion,
             device,
             epoch,
             statistics_callback=None):
    """Validate the model."""
    statistics_callback = statistics_callback or (lambda x: None)

    with torch.no_grad():
        model.eval()
        progress = tqdm.tqdm(val_loader, desc="Validation Batch")

        accumulated_loss = 0
        accumulated_miou = 0

        mious = []

        for batch_index, batch in enumerate(progress):
            source_batch = batch['image'].to(device)
            target_batch = batch['label'].to(device)

            output_batch = model(source_batch)
            loss = criterion(output_batch, target_batch)
            batch_mious = list(calculate_many_mious(output_batch, target_batch))
            miou = np.array(batch_mious).mean()
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
            mious += batch_mious

    return accumulated_loss, accumulated_miou, mious


def training_loop(model,
                  train_loader,
                  val_loader,
                  criterion,
                  optimizer,
                  scheduler,
                  device,
                  epochs,
                  statistics_callback=None,
                  epoch_end_callback=None,
                  start_epoch=0):
    """The main training loop."""
    epoch_end_callback = epoch_end_callback or (lambda x: None)
    statistics_callback = statistics_callback or (lambda x: None)

    for epoch in tqdm.tqdm(range(start_epoch, epochs), desc="Epoch"):
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

        accumulated_loss, accumulated_miou, mious = validate(model,
                                                             val_loader,
                                                             criterion,
                                                             device,
                                                             epoch,
                                                             statistics_callback)

        tqdm.tqdm.write("{}: Epoch {}, Validation loss: {}, Validation mIoU: {}".format(os.environ.get("EXPERIMENT", "segmentation"),
                                                                                        epoch,
                                                                                        accumulated_loss / len(val_loader),
                                                                                        accumulated_miou / len(val_loader)))


        # Now that we have all mious, we can find the indices of "interesting" ones (eg, ones
        # that did well, ones that did badly...
        epoch_end_callback(model, optimizer, val_loader, np.array(mious), epoch)



def call_many(*functions):
    def _inner(*args, **kwargs):
        return [f(*args, **kwargs) for f in functions]

    return _inner


def create_functor_for_segmenting_image(dataset, image_callback, device, *args, **kwargs):
    """Create a functor for segmenting the image based on its position in the dataset."""
    def create_functor_for_image(i):
        """Segment a given image from the dataset."""
        input_pair = dataset_with_deterministic_transforms[i]
        viewable_pair = dataset_with_viewable_transforms[i]
        image = input_pair["image"].to(device)
        viewable_image = viewable_pair["image"].to(device)
        viewable_label = viewable_pair["label"].to(device)
        palette = viewable_pair["label_palette"]

        return image_callback(i,
                              image,
                              viewable_image,
                              viewable_label,
                              palette=palette,
                              *args,
                              **kwargs)


    dataset_with_deterministic_transforms = dataset.with_deterministic_transforms()
    dataset_with_viewable_transforms = dataset.with_viewable_transforms()
    return create_functor_for_image


def save_segmentations_for_first_n_images(model, dataset, path, n, device):
    """Create functions to save segmentations for the first n images."""
    def on_received_image(i,
                          image,
                          viewable_image,
                          viewable_label,
                          palette=None,
                          *args,
                          **kwargs):
        return save_segmentations_for_image(model,
                                            viewable_image,
                                            image,
                                            viewable_label,
                                            os.path.join(path, "image_{}.png".format(i)),
                                            palette=palette)


    functor_factory = create_functor_for_segmenting_image(dataset,
                                                          on_received_image,
                                                          device)

    return [functor_factory(i) for i in range(0, n)]


def segment_and_tensorify(model, image, palette=None):
    """Segment an image, then convert predictions to tensor."""
    pred, output = segment_image(model, input)
    return torch.tensor(pred), output


def viewable_image_label_pair_to_images(viewable_image, viewable_label, palette=None):
    """Convert the viewable_image and viewable_label pair into actual Image instances."""
    label_image = segmentation_to_image(viewable_label.cpu().numpy(),
                                        palette=palette)
    tensor_image = Image.fromarray(viewable_image.cpu().numpy().transpose(1, 2, 0).astype('uint8'))

    return label_image, tensor_image


def segment_and_produce_tensorboard_image(model, image, label, palette=None):
    """Segment image and produce a tensorboard-loggable image and miou."""
    pred, output_tensor = segment_image(model, image)
    miou = calculate_miou(output_tensor.detach(), label)
    segmentation_image = segmentation_to_image(pred,
                                               palette=palette)

    return segmentation_image, miou


def write_first_n_images_to_tensorboard(model, dataset, summary_writer, n, device, set_name=None):
    """Create functions to save segmentations for the first n images."""
    set_name = set_name or "validation"

    def on_received_image(i,
                          image,
                          viewable_image,
                          viewable_label,
                          palette=None,
                          *args,
                          **kwargs):
        def on_received_statistics(statistics):
            """We got the statistics, now we can segment the image."""
            if statistics["mode"] != "validation" or statistics["batch_index"] != 0:
                return

            segmentation_image, miou = segment_and_produce_tensorboard_image(model.to(device),
                                                                             image.to(device),
                                                                             viewable_label,
                                                                             palette=palette)

            # Blend segmentation on top of real image
            overlay_segmentation_image = overlay_segmentation(tensor_image, segmentation_image)

            summary_writer.add_images("{}/reference/{}".format(set_name, i), np.hstack([
                np.asarray(overlay_label_image.convert('RGB')),
                np.asarray(overlay_segmentation_image.convert('RGB'))
            ]), global_step=statistics["epoch"], dataformats='HWC')
            summary_writer.add_scalar("{}/reference/{}/mIoU".format(set_name, i),
                                      miou,
                                      global_step=statistics["epoch"])

        label_image, tensor_image = viewable_image_label_pair_to_images(viewable_image,
                                                                        viewable_label)

        # With the label and the image, we can blend the segmentation
        # and label on top of the original image
        overlay_label_image = overlay_segmentation(tensor_image, label_image)

        return on_received_statistics


    functor_factory = create_functor_for_segmenting_image(dataset,
                                                          on_received_image,
                                                          device)

    return [functor_factory(i) for i in range(0, n)]


def save_interesting_images_to_tensorboard(summary_writer,
                                           device):
    """Save some interesting images from the validation process on each epoch.

    'Interesting' is defined as the three images with the best segmentation,
    the three images with the worst segmentation and the three images in the
    middle.
    """
    def save_pairs(model, pairs, epoch, tag):
        """Helper to save pairs of images."""
        for i, pair in enumerate(pairs):
            network_input, viewable = pair
            viewable_image = viewable["image"]
            viewable_label = viewable["label"]
            palette = network_input["label_palette"]

            label_image, tensor_image = viewable_image_label_pair_to_images(viewable_image,
                                                                            viewable_label)

            # With the label and the image, we can blend the segmentation
            # and label on top of the original image
            overlay_label_image = overlay_segmentation(tensor_image, label_image)

            segmentation_image, miou = segment_and_produce_tensorboard_image(model,
                                                                             network_input["image"].to(device),
                                                                             viewable_label.to(device),
                                                                             palette=palette)

            # Blend segmentation on top of real image
            overlay_segmentation_image = overlay_segmentation(tensor_image, segmentation_image)

            summary_writer.add_images("validation/{}/{}".format(tag, i), np.hstack([
                np.asarray(overlay_label_image.convert('RGB')),
                np.asarray(overlay_segmentation_image.convert('RGB'))
            ]), global_step=epoch, dataformats='HWC')
            summary_writer.add_scalar("validation/{}/{}/mIoU".format(tag, i),
                                      miou,
                                      global_step=epoch)


    def _inner(model, optimizer, val_loader, mious, epoch):
        ordered_mious = mious.argsort()
        viewable_val_loader = val_loader.dataset.with_viewable_transforms()

        length = len(ordered_mious)
        middle = (length - 1) // 2
        worst_three = [(val_loader.dataset[i], viewable_val_loader[i]) for i in ordered_mious[:3]]
        best_three = [(val_loader.dataset[i], viewable_val_loader[i]) for i in ordered_mious[-3:]]
        middle_three = [(val_loader.dataset[i], viewable_val_loader[i]) for i in ordered_mious[middle - 1:middle + 1]]

        # Now that we have the worst, best and middle images,
        # save them
        save_pairs(model, worst_three, epoch, "worst")
        save_pairs(model, best_three, epoch, "best")
        save_pairs(model, middle_three, epoch, "middle")

    return _inner


def update_tensorboard_logs(summary_writer):
    """A function to write new logs to tensorboard."""
    def on_statistics_available(statistics):
        summary_writer.add_scalar("{}/loss".format(statistics["mode"]),
                                  statistics["statistics"]["loss"],
                                  statistics["epoch"])
        summary_writer.add_scalar("{}/mIoU".format(statistics["mode"]),
                                  statistics["statistics"]["mIoU"],
                                  statistics["epoch"])

    return on_statistics_available


def read_classes_trainval(classes_trainval):
    """Read the trainval file and output list of strings."""
    with open(classes_trainval, "r") as f:
        return [l.split()[0] for l in f.readlines() if l.split()[1] != "-1"]


def read_classes_list(classes_directory):
    """For each class in the classes directory, list image identifiers per class."""
    for filename in fnmatch.filter(os.listdir(classes_directory), "*_trainval.txt"):
        image_class = filename.split("_")[0]
        yield (image_class,
               read_classes_trainval(os.path.join(classes_directory, filename)))



def add_schedule(cls, query_step, max_steps):
    class ScheduledDropoutLayer(nn.Module):
        def __init__(self, p=1.0):
            super().__init__()
            self.dropout_cls = cls
            self.dropout = self.dropout_cls(p=1.0)
            self.current_step = -1
            self.query_steps = query_step
            self.max_steps = max_steps
            self.p = p

        def forward(self, x):
            """Go forward, altering dropout layer if necessary."""
            if self.training:
                this_step = self.query_steps()
                if this_step != self.current_step:
                    self.current_step = this_step
                    alpha = self.current_step / self.max_steps
                    new_p = min((0 * (1.0 - alpha) + self.p * (alpha)), self.p)
                    self.dropout = self.dropout_cls(p=new_p)
                    tqdm.tqdm.write("Update dropout p to {} (max {})".format(new_p, self.p))

            return self.dropout(x)
    
    return ScheduledDropoutLayer


class ChannelUOut(nn.Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p
        self.dist = torch.distributions.uniform.Uniform(-p, p)
    
    def forward(x):
        if self.training:
            return x

        _, c, __, ___ = x.shape
        channel_samples = self.dist.sample(c).view(1, c)

        return x + x * channel_samples


def create_drop_layer_cls(layer_type, query_step, max_steps):
    """Create a Dropout layer depending on the type."""
    if layer_type == "channel":
        return add_schedule(nn.Dropout2d, query_step, max_steps)
    elif layer_type == "channel-uout":
        return add_schedule(ChannelUOut, query_step, max_steps)
    elif layer_type == "patch":
        return add_schedule(DropoutActivationPatch, query_step, max_steps)


class ScheduleTracker(object):
    """Just keeps track of which epoch it is."""
    def __init__(self, initial_epoch):
        super().__init__()
        self.epoch = initial_epoch

    def query(self):
        return self.epoch

    def epoch_end(self):
        self.epoch += 1


def create_model(input_channels, args, schedule_tracker):
    """Create model from the arguments."""
    if args.model == "deeplabv3":
        model = DeepLabModel(input_channels=3,
                             num_classes=args.num_classes,
                             drop_layer_cls=create_drop_layer_cls(args.drop_type, schedule_tracker.query, args.full_dropout_epoch),
                             feature_detection_dropout_rate=args.feature_detection_dropout_rate,
                             pyramid_dropout_rate=args.pyramid_dropout_rate,
                             decoder_dropout_rate=args.decoder_dropout_rate,
                             decoder_use_channel_attention=args.decoder_use_channel_attention)
    elif args.model == "modelzoo-deeplabv3":
        zoo_model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
        model = DeepLabModelZooModel(zoo_model.backbone, zoo_model.classifier)
        del zoo_model

    return model


class FocalLoss(nn.Module):
    """FocalLoss implementation.

    Essentially we don't penalize a model for being reasonably confident
    about a class prediction, even if it is not perfectly confident. This
    means that we pay attention to other gradients, which may help to
    improve overall classification.
    """

    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = remap_tensor_digit(target.view(-1,1), { 255: 0 })

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1).long())
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss * torch.tensor([1.0 if t != self.ignore_index else 0.0 for t in target.view(-1).cpu().numpy()],
                                   device=loss.device,
                                   dtype=torch.float32)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def remap_tensor_digit(tensor, digit_map):
    """Remap the digit in tensor."""
    return torch.tensor([
        digit_map[int(d)] if int(d) in digit_map else int(d) for d in tensor.cpu().numpy().ravel().tolist()
    ], dtype=torch.long, device=tensor.device).view(tensor.shape)


def dice_loss(eps=1e-7, ignore_index=255):
    """Computes the Sørensen–Dice loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    def inner(logits, true):
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[remap_tensor_digit(true.squeeze(1), { 255: 0 })]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)
    return inner



LOSS_FUNCTIONS = {
    "cross-entropy": segmentation_cross_entropy_loss(size_average=None, ignore_index=255),
    "focal-loss": FocalLoss(gamma=2, size_average=True, ignore_index=255),
    "dice-loss": dice_loss(ignore_index=255)
}



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
    parser.add_arguemnt("--full-dropout-epoch", type=int, default=30, help="Number of epochs until dropout is effective")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--drop-type", type=str, choices=("channel", "patch", "channel-uout"), default="channel", help="Dropout type")
    parser.add_argument("--feature-detection-dropout-rate",
                        type=float,
                        default=0.0,
                        help="Dropout rate in feature detection layers")
    parser.add_argument("--decoder-dropout-rate",
                        type=float,
                        default=0.0,
                        help="Dropout rate for Decoder layers.")
    parser.add_argument("--decoder-use-channel-attention",
                        action='store_true',
                        default=False,
                        help="Use ChannelAttention in Decoder layers.")
    parser.add_argument("--pyramid-dropout-rate",
                        type=float,
                        default=0.0,
                        help="Dropout rate for Pyramid Pooling layers.")
    parser.add_argument("--num-classes", type=int, default=21, help="Number of segmentation classes")
    parser.add_argument("--save-to", type=str, help="Where to save the model to", required=True)
    parser.add_argument("--log-statistics", type=str, help="Where to log statistics to", default="logs/statistics")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--test-only", action="store_true", help="Only test")
    parser.add_argument("--load-from", type=str, help="Where to load the model from")
    parser.add_argument("--save-interesting-images", type=str, help="Where to save interesting images", default="logs")
    parser.add_argument("--model",
                        type=str,
                        help="Which model to create",
                        default="deeplabv3",
                        choices=["deeplabv3", "modelzoo-deeplabv3"])
    parser.add_argument("--limit-train-data", type=float, default=1.0, help="Percentage of training data to use")
    parser.add_argument("--limit-validation-data", type=float, default=1.0, help="Percentage of validation data to use")
    parser.add_argument("--classes-list", type=str, help="Path to image classes list")
    parser.add_argument("--loss-function", type=str,
                        help="Loss Function to use",
                        choices=("cross-entropy", "focal-loss", "dice-loss"),
                        default="cross-entropy")
    args = parser.parse_args()

    (train_loader,
     val_loader,
     test_loader) = load_data(args.source_images,
                              args.segmentation_images,
                              args.training_set,
                              args.validation_set,
                              args.test_set,
                              batch_size=args.batch_size,
                              limit_train_data=args.limit_train_data,
                              limit_validation_data=args.limit_validation_data,
                              classes_list=dict(read_classes_list(args.classes_list)))
    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    schedule_tracker = ScheduleTracker(0)
    model = create_model(input_channels=3, args=args, schedule_tracker=schedule_tracker).to(device)

    print(model)
    criterion = LOSS_FUNCTIONS[args.loss_function]
    optimizer = optim.SGD(differential_learning_rates(model,
                                                      model.optimizer_params(),
                                                      args.learning_rate),
                          momentum=0.9,
                          weight_decay=5e-4,
                          nesterov=False)
    start_epoch = 0
    best_accumulated_miou = 0

    if args.load_from:
        saved_info = torch.load(args.load_from)
        start_epoch = saved_info['epoch']
        best_accumulated_miou = saved_info['best_pred']
        optimizer.load_state_dict(saved_info['optimizer'])
        model.load_state_dict(saved_info['state_dict'])
        schedule_tracker.epoch = start_epoch
        print("Loaded model from {}".format(args.load_from))

    scheduler = PolynomialLearningRateScheduler(optimizer,
                                                args.learning_rate,
                                                start_epoch + args.epochs,
                                                len(train_loader))
    writer = SummaryWriter(os.path.join(os.path.dirname(args.log_statistics), "tensorboard"))

    if not args.test_only:
        training_loop(model,
                      train_loader,
                      val_loader,
                      criterion,
                      optimizer,
                      scheduler,
                      device,
                      epochs=args.epochs,
                      statistics_callback=call_many(
                          log_statistics(args.log_statistics),
                          update_tensorboard_logs(writer),
                          # Take the first image from the first three batches
                          *list(itertools.chain.from_iterable((
                               write_first_n_images_to_tensorboard(model,
                                                                   val_loader.dataset,
                                                                   writer,
                                                                   3,
                                                                   device,
                                                                   set_name="validation"),
                               write_first_n_images_to_tensorboard(model,
                                                                   train_loader.dataset,
                                                                   writer,
                                                                   3,
                                                                   device,
                                                                   set_name="train"))
                         ))
                      ),
                      epoch_end_callback=call_many(
                          save_model_on_better_miou(args.save_to,
                                                    best_accumulated_miou),
                          save_interesting_images_to_tensorboard(writer, device),
                          lambda m, p, e, t: schedule_tracker.epoch_end()
                      ),
                      start_epoch=start_epoch)

    tqdm.tqdm.write("Performing final validation set pass")
    accumulated_loss, accumulated_miou, mious = validate(model,
                                                         test_loader,
                                                         criterion,
                                                         device,
                                                         args.epochs,
                                                         statistics_callback=call_many(
                                                             update_tensorboard_logs(writer),
                                                             log_statistics(args.log_statistics),
                                                             # Take the first image from the first three batches
                                                             *list(itertools.chain.from_iterable([
                                                                  write_first_n_images_to_tensorboard(model,
                                                                                                       val_loader.dataset,
                                                                                                       writer,
                                                                                                       3,
                                                                                                       device,
                                                                                                       set_name="validation"),
                                                                   write_first_n_images_to_tensorboard(model,
                                                                                                       train_loader.dataset,
                                                                                                       writer,
                                                                                                       3,
                                                                                                       device,
                                                                                                       set_name="train")
                                                            ]))
                                                         ))

    save_interesting_images(os.path.join(args.save_interesting_images,
                                         "interesting",
                                         "image.png"),
                            device)(model, optimizer, test_loader, np.array(mious), args.epochs)


if __name__ == "__main__":
    main()
