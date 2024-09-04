# Crack Propagation Prediciton using UNet

This project aims to develop a deep learning model to predict crack propagation in materials using a UNet architecture. The project leverages simulation data and material properties to simulate and predict crack growth in solid structures. The model can be trained, evaluated, and applied to new data for predicting crack growth patterns based on material characteristics and stress conditions.

## Project Overview

### Key Components:
1. **Data Preprocessing and Conversion**:  
   Converts raw data from material simulations into TensorFlow-compatible TFRecord format, ready for training and evaluation.
   
2. **Model Training**:  
   A UNet architecture is used to predict crack growth. The model is trained using Keras, TensorFlow, and CIDS, with options for multi-GPU training and automatic logging of metrics using Weights & Biases (WandB).
   
3. **Inference/Prediction**:  
   The trained model is used to predict crack growth in new simulations. Predictions are compared with the ground truth, and various performance metrics (e.g., MSE, MAE, SSIM) are calculated.

### Data
The data consists of multiple input features, including:
- **Material stress, energy, and displacement fields**
- **Anisotropy angles**
- **Material anisotropy**

These inputs are provided as images or tensors, with corresponding outputs representing the crack propagation state.