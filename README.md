# Convolutional neural networks as a chess heuristic

This repository contains a Python script for training and using a Convolutional Neural Network (CNN) to predict the heuristic value of a chess position in depth 15.

Prerequisites
Before using this code, ensure you have the following dependencies installed:

- Pandas: A library for data manipulation and analysis.
- NumPy: A library for numerical computing.
- TensorFlow: An open-source machine learning framework.
- Matplotlib: A library for creating visualizations.
- scikit-learn: A library for machine learning and data preprocessing.
You can install these dependencies using pip.

```
pip install pandas numpy tensorflow matplotlib scikit-learn
```


## Data Preparation
The script loads the dataset from the MatrizPosiciones.json file and splits it into training and testing sets. It then normalizes the data for training.

The MatrizPosiciones.json contains the evaluations in depth 15 using stockfish 16, it splits each chess position in several matrix that contains each of the pieces, for example, imagine a chess position with a pawn and a king, the json will contain the full chess position as a matrix, another matrix with only the pawn and another matrix with the king, the values for the pieces are:

- Pawns: 1, -1
- Knights: 2, -2
- Bishops: 3, -3
- Rooks: 4, -4
- Queens: 5, -5
- Kings: 6, -6

## CNN Model
The CNN model used for this task consists of convolutional layers followed by max-pooling layers and dense layers. The model architecture is as follows:

- Input Layer: 128x8x8 (128 channels, 8x8 pixels)
- Convolutional Layer: 128 filters, 2x2 kernel, ReLU activation
- Max-Pooling Layer: 1x1 pool size
- Convolutional Layer: 256 filters, 2x2 kernel, ReLU activation
- Max-Pooling Layer: 1x1 pool size
- Convolutional Layer: 256 filters, 2x2 kernel, ReLU activation
- Max-Pooling Layer: 1x1 pool size
- Convolutional Layer: 256 filters, 2x2 kernel, ReLU activation
- Flatten Layer
- Fully Connected Layer: 256 units, ReLU activation
- Output Layer: 1 unit

## Training results

The results of the training at 60 EPOCHS are the following:

![training](https://github.com/nenomg/Chess-convolutional-neural-networks-as-a-heuristic/assets/105873794/3a8662bc-ed7b-4bde-a52e-b9fcb5dffe0e)

![trainingLoss](https://github.com/nenomg/Chess-convolutional-neural-networks-as-a-heuristic/assets/105873794/a336a7a8-1c3b-4089-810a-07431fc94743)


## Conclusions

It is possible to make a model that learns from another engine, so it will be possible to make a reinforcement learning model using convolutional neural networks.
