#!/bin/bash

EXPERIMENT=${EXPERIMENT:-default}

python train_semantic_segmentation.py \
    --cuda \
    --source-images ./data/VOCdevkit/VOC2012/JPEGImages \
    --segmentation-images ./data/VOCdevkit/VOC2012/SegmentationClass \
    --training-set ./data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt \
    --validation-set ./data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt \
    --test-set ./data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt \
    --save-to experiments/$EXPERIMENT/model.pt \
    --log-statistics experiments/$EXPERIMENT/logs/statistics \
    --save-interesting-images experiments/$EXPERIMENT/logs/interesting \
    $@
