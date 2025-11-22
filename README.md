# DermAI – Grad-CAM Explainability Module

This module provides visual interpretability for the DermAI skin lesion classification model using Gradient-weighted Class Activation Mapping (Grad-CAM). It supports transparency in clinical decision support systems by showing which regions of the image contributed most to the model’s prediction.

## Overview

After completing the model training and evaluation phases in DermAI, the best-performing CNN model was exported for deployment.  
This module loads that trained model and generates Grad-CAM visual explanations for both benign and malignant predictions.  
It is implemented as an independent component that can be integrated into the backend during inference, or invoked upon user request when an explanation is needed.

## Features

- Load the pre-trained CNN model 
- Preprocess input images using the same pipeline as training
- Identify and extract the last convolutional layer activations
- Compute gradients for the predicted class
- Generate Grad-CAM heatmaps
- Overlay heatmaps on the original image
- Export results as image files or base64 for API usage
- Fully compatible with integration into DermAI's backend service

## Folder Structure
_____________

## Requirements

The module depends on the following libraries:

- TensorFlow / Keras  
- NumPy  
- OpenCV-Python  
- Matplotlib  
- Pillow (optional)



