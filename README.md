# CaptionCraft

## Overview

In today's digital era, effective communication of visual content remains challenging. This project introduces an automated image caption generator using deep learning techniques to bridge the gap between imagery and textual description.

## Key Features

- **Hybrid Architecture**: Utilizes VGG16, ResNet-50, and Transformer models for feature extraction and caption generation.
- **Performance Evaluation**: Compares models based on BLEU, METEOR, and ROUGE metrics using the Flickr8K dataset.
- **Frontend Interface**: Provides a user-friendly interface for generating captions from uploaded images.
- **Modular Design**: Enables easy deployment across platforms and applications.

## Dataset

Trained on the Flickr8K dataset:
- 8,091 images with descriptive captions.
- Ensures comprehensive training and evaluation.
- https://www.kaggle.com/datasets/adityajn105/flickr8k

## Methodology

### Models Trained

1. **VGG16**: Extracts hierarchical visual representations.
2. **ResNet-50**: Focuses on object categories and spatial relationships.
3. **Transformer**: Implements attention-based mechanisms for language modeling.

### Training Approach

- **CNN Encoder (VGG16, ResNet-50)**: Extracts semantic features.
- **Transformer Encoder**: Captures global and local features simultaneously.
- **LSTM Decoder**: Generates natural language descriptions.
- **Evaluation**: Uses Adam optimizer, categorical cross-entropy loss, teacher forcing, and beam search.

## Frontend Interface

- **User Interaction**: Upload images, view generated captions.
- **Integration**: Connects to backend using Flask for model inference.
- **Frameworks**: HTML, CSS, JavaScript .
![image](https://github.com/user-attachments/assets/9a733231-848a-4e83-9f42-58c308361ecb)

## Research Paper

https://1drv.ms/w/c/5573114cc8880c16/QRYMiMhMEXMggFXGBgAAAAAAM23x0h2NdW24ow





