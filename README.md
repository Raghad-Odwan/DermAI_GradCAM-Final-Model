# DermAI – Final Model and Grad-CAM Module

## Overview

This repository contains the **final trained deep learning model** used in the DermAI system, along with the implementation of **Grad-CAM explainability** for skin lesion image classification.

The repository represents the final stage of the AI pipeline, focusing on model inference and visual explanation rather than experimentation or training.

This work is intended for academic and research purposes only.

---

## Repository Structure

```
DermAI_GradCAM-Final-Model/
├── Model/
│   └── model_link.md
│
├── Model_code/
│   ├── inference.py
│   ├── gradcam.py
│   └── utils.py
│
├── notebooks/
│   └── GradCAM_Demo.ipynb
│
└── README.md
```

---

## Final Model Access

The final trained model weights are stored externally due to file size constraints.

The model can be accessed through the following link:

See `Model/model_link.md`

---

## Model Description

* Architecture: ResNet50 (Transfer Learning)
* Input Size: 224 × 224 RGB images
* Output: Binary classification (Benign / Malignant)
* Training Strategy: Transfer learning with data augmentation and class weighting
* Evaluation: Cross-validation and final test set evaluation

---

## Inference Usage

The model can be used for inference as follows:

```python
from Model_code.inference import predict

image_path = "path_to_image.jpg"
result = predict(image_path)

print(result)
```

The returned result includes the predicted class label and confidence score.

---

## Grad-CAM Explainability

This repository includes Grad-CAM functionality to visualize the regions of the image that contribute most to the model’s prediction.

Example usage:

```python
from Model_code.gradcam import generate_gradcam

image_path = "path_to_image.jpg"
output_path = "path_to_gradcam.jpg"

generate_gradcam(image_path, output_path)
```
---

## System Integration

This module is designed to be integrated with the DermAI backend system.
It receives validated images from the image validation module and returns both prediction results and explainability outputs.

---


أخبرني.
