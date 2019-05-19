# DeepLab v3+ with Channel Attention and Dropout
This repository is a reimplementation of https://github.com/jfzhang95/pytorch-deeplab-xception with two key changes - Channel Attention and Channel Dropout.

(By @smspillaz and @PaavoLeonhard).

We noticed that the original DeepLabv3+ [paper](https://arxiv.org/abs/1802.02611) made no mention of regularization techniques. We suspect that due to the large number of channels that are present in the model, it is possible that the network could over-weighting certain channels on the training set, meaning that the validation and test set performance could be lower. We therefore suggest two regularization mechanisms, Channel Attention and Channel Dropout to ameliorate this effect. We find that where overfitting is forced by training with a small training set that appropriate dropout and attention has a substantial regularizing effect.

# Model Architecture

![Overall Architecture of the Model](https://docs.google.com/drawings/d/e/2PACX-1vT3zJFeRkmVcH1lSjj6XLsyL96V-fWgWOEQtMtZry3geSeZZK0qxajKKFgSDUxRZDpQ-CsbYqgLFf1D/pub?w=1713&h=1273)

# Performance
## VOC2012, full dataset

| Experiment | Train Loss | Train mIoU | Val Loss | Val mIoU |
|------------|------------|------------|----------|----------|
|Baseline    | 0.01       | 0.88       |0.03      | 0.76     |
|ResNet Dropout2D    | 0.02       | 0.65       |0.03      | 0.69     |
|Pyramid Pooling Dropout2D    | 0.01       | 0.88       |0.03      | 0.75     |
|Decoder Dropout2D    | 0.01       | 0.88       |0.03      | 0.76     |
|Decoder ChannelAttention    | 0.01       | 0.81       |0.03      | 0.75     |
|All    | 0.03       | 0.64       |0.04      | 0.67     |

## VOC2012, 10 train images

| Experiment | Train Loss | Train mIoU | Val Loss | Val mIoU |
|------------|------------|------------|----------|----------|
|Baseline    | 0.01       | 0.82       |0.19      | 0.38     |
|ResNet Dropout2D    | 0.08       | 0.51       |0.47      | 0.47     |
|Pyramid Pooling Dropout2D    | 0.01       | 0.17       |0.03      | 0.42     |
|Decoder Dropout2D    | 0.01       | 0.89       |0.17      | 0.4     |
|Decoder ChannelAttention    | 0.01       | 0.86       |0.18      | 0.36     |
|Decoder ChannelAttention and Dropout2D    | 0.01       | 0.18       |0.03      | 0.43     |
|All    | 0.02       | 0.71       |0.22      | 0.39     |
