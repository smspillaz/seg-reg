# DeepLab v3+ with Channel Attention and Dropout
This repository is a reimplementation of https://github.com/jfzhang95/pytorch-deeplab-xception with two key changes - Channel Attention and Channel Dropout.

(By @smspillaz and @PaavoLeonhard).

We noticed that the original DeepLabv3+ [paper](https://arxiv.org/abs/1802.02611) made no mention of regularization techniques. We suspect that due to the large number of channels that are present in the model, it is possible that the network could over-weighting certain channels on the training set, meaning that the validation and test set performance could be lower. We therefore suggest two regularization mechanisms, Channel Attention and Channel Dropout to ameliorate this effect. We find that where overfitting is forced by training with a small training set that appropriate dropout and attention has a substantial regularizing effect.

# Model Architecture

![Overall Architecture of the Model](https://docs.google.com/drawings/d/e/2PACX-1vT3zJFeRkmVcH1lSjj6XLsyL96V-fWgWOEQtMtZry3geSeZZK0qxajKKFgSDUxRZDpQ-CsbYqgLFf1D/pub?w=1713&h=1273)
