# Melanoma-Detection-Using-Skin-Lesion-Trained-Models

## Project Overview

This repository contains the code and report for the project "Melanoma Detection using Skin Lesion-Trained Models". The project utilizes deep learning models to detect melanoma from skin lesion images. The models used include pre-trained architectures such as VGG19, ResNet50, EfficientNetV2, and SwinNetV2, as well as a custom CNN model.

## Introduction

Skin cancer, particularly melanoma, poses a significant health risk globally. Early detection is crucial for successful treatment. This project aims to leverage deep learning for accurate melanoma detection, aiding dermatologists in diagnosis.

## Dataset

The dataset used is the ISIC-2019 dataset, a comprehensive collection of skin lesion images for research in melanoma detection and classification. The dataset can be downloaded from Kaggle.

Note: The ISIC_2019_training_input folder is not included in this repository due to its large size. Please download it directly from Kaggle.

## Models

We explored the effectiveness of transfer learning on the following architectures:

**VGG19:** Known for its simplicity and uniformity with 19 layers.

**ResNet50:** A 50-layer model introducing residual learning.

**EfficientNetV2:** Known for achieving high accuracies with fewer parameters.

**SwinNetV2:** Incorporates hierarchical transformer layers for capturing long-range dependencies.

**Custom CNN Model:** A custom architecture tailored for melanoma detection.

## Data Preprocessing

The dataset was divided into training (80%) and testing (20%) sets. Due to the imbalanced nature of the dataset, data augmentation techniques were applied to balance the classes during training. The augmentation strategies varied based on the label classes.

### Augmentation Techniques

**Category 1 (NV):** Normalization, horizontal and vertical flipping.

**Category 2 (MEL, BCC, BKL):** Normalization, flipping, and small rotations.

**Category 3 (AK, SCC, VASC, DF):** Normalization, flipping, rotations, color jitter, and affine transformations.

To handle the label-based data augmentation, custom collate functions were created for training, validation, and testing.

## Training and Evaluation

Transfer learning was implemented using PyTorch and the torchvision.models and timm libraries. Models were trained with CrossEntropy Loss and the SGD optimizer. A learning rate scheduler was used to improve training efficiency.

### Cross-Validation

We performed 5-fold cross-validation to ensure robustness. Each fold involved training and validation on different splits of the dataset.

### Model Training

We employed the CrossEntropy Loss function, which is commonly used for multi-class classification tasks, to calculate the loss during training. We used the SGD optimizer with a learning rate of 0.001 and momentum of 0.9 to update the model parameters. To improve training efficiency and convergence, we used PyTorch’s learning rate scheduler with a step size of 1 and a gamma of 0.1. If the performance of the model worsens, the scheduler reduces the learning rate by a factor of 10 on every epoch.

For the vanilla 80-20 split of data into train/validation and test sets, we trained the pre-trained models for 2 epochs and our custom model for 4 epochs. For cross-fold validation split, we trained the transfer learning models for 1 epoch (5 folds) and our custom model for 2 epochs (5 folds per epoch).

### Model Evaluation

After training the models, we evaluated their performance on the validation set and the test set. We calculated metrics such as train/val/test accuracy, train/val/test loss, precision, recall, F1-score, and AUC-ROC to assess the models’ performance.

## Results

The models were evaluated based on accuracy, precision, recall, F1-score, and AUC-ROC. The custom model performed best in terms of overall accuracy, while SwinNetV2 achieved the highest recall, indicating its effectiveness in minimizing false negatives.

### Key Findings

**VGG19:** Best testing accuracy in the initial split.

**Custom Model:** Best performance in cross-validation.

**SwinNetV2:** Highest recall, important for minimizing false negatives.

### Detailed Results

**VGG19:** Provided a good baseline with simple 3x3 convolutional filters and max-pooling layers.

**ResNet50:** Demonstrated how residual learning and skip connections can affect performance.

**EfficientNetV2:** Showed remarkable accuracies with fewer parameters, making it suitable for tasks with limited computational resources.

**SwinNetV2:** Efficiently captured long-range dependencies with shifted windows and hierarchical transformer layers.

**Custom Model:** Tailored specifically for the dataset, demonstrated good generalization and performance.

## Files in the Repository

**Group5_Final_project_report.pdf:** Detailed project report including methodology, results, and discussion.

**ISIC_2019_Training_GroundTruth.csv:** Ground truth labels for the ISIC-2019 dataset.

**ISIC_2019_Training_Metadata.csv:** Metadata for the ISIC-2019 dataset.

**custom.ipynb:** Jupyter notebook for the custom CNN model.

**EfficientNetV2_final.ipynb:** Jupyter notebook for the EfficientNetV2 model.

**resnet50.ipynb:** Jupyter notebook for the ResNet50 model.

**SwinNetV2.ipynb:** Jupyter notebook for the SwinNetV2 model.

**vgg19.ipynb:** Jupyter notebook for the VGG19 model.
