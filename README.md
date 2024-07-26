Stock Market Prediction with PyTorch using LSTM
=============================================================

This project aims to predict the stock market trends for Remedy Company using a Long Short-Term Memory (LSTM) neural network using PyTorch. The model is trained on historical stock data and uses a combination of technical indicators to make predictions.

## Dataset
The dataset used in this project is obtained from Yahoo Finance and consists of historical stock data for Remedy Company (REMEDY.HE). The dataset is preprocessed using a custom pipeline to transform the data into a format suitable for training an LSTM model.

## Model Architecture
The LSTM model used in this project consists of an input layer, four LSTM layers, and an output layer. The input layer takes in sequences of length 7, which are then fed into the LSTM layers. The output layer produces a single value representing the predicted stock price.

## Training and Evaluation
The model is trained using a combination of mean squared error (MSE) as the loss function and Adam optimizer. The training process is divided into epochs, where each epoch consists of a forward pass, backward pass, and optimization step. The model is evaluated using the mean absolute error (MAE) metric.

## Pipelines and Utilities
The project uses several custom pipelines and utilities to preprocess the data and train the model. These include:

* **DropColumns**: A pipeline to drop unnecessary columns from the dataset.
* **Scaler**: A pipeline to scale the data using Min-Max Scaler.
* **SequencePipeline**: A pipeline to transform the data into sequences of length 7.
* **MakeSequence**: A utility to create sequences from the dataset.
* **train_test_split**: A utility to split the dataset into training and testing sets.

## Results

### Loss Curves

The training and testing loss curves are shown below:

<img src="./images/loss_curve.png" alt="Experiment_Loss_Curve" style="width: 50%; height: auto; margin: 0 auto; display: block;" />

### Predictions

The predicted close prices for the testing set are shown below:

<img src="./images/experiment_test_prediction.png" alt="Experiment Test Prediction" style="width: 50%; height: auto; margin: 0 auto; display: block;" />

## Usage
To use this project, simply clone the repository and run the `main.py` file. This will preprocess the data, train the model, and evaluate its performance.
Or you can just **check notebooks** files for more detailed explanation.

## Requirements
* Python 3.8+
* PyTorch 1.9+
* Pandas 1.3+
* NumPy 1.20+
* Scikit-learn 0.24+