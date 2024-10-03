# CNN with Attention Mechanism on MNIST

This project demonstrates the use of an attention mechanism in a Convolutional Neural Network (CNN) to classify images from the MNIST dataset. The attention mechanism helps the model focus on important parts of the input image, potentially improving classification performance.

## Overview

The project consists of a CNN model with an integrated attention mechanism. The attention mechanism is implemented as a separate module that computes attention weights for the input feature maps. These weights are then used to emphasize important features before further processing.

## Model Architecture

The model architecture includes:

- Two convolutional layers for feature extraction.
- An attention mechanism that computes attention weights for the feature maps.
- A fully connected layer for classification.

The attention mechanism is defined as follows:

## Visualize
| Label 1 | Label 3 | Label 4 |
|:-------:|:-------:|:-------:|
| ![Label 1](images/label_1.png) | ![Label 3](images/label_3.png) | ![Label 4](images/label_4.png) |

| Label 6 | Label 7 | Label 9 |
|:-------:|:-------:|:-------:|
| ![Label 6](images/label_6.png) | ![Label 7](images/label_7.png) | ![Label 9](images/label_9.png) |

## Usage

```
pip install -r requirements.txt
```

```
python main.py
```