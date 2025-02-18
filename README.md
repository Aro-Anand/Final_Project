# Waste Classification using Convolutional Neural Networks

An end-to-end solution for automated waste classification using a custom Convolutional Neural Network (CNN) and Streamlit for deployment. This project aims to address the challenges of manual waste sorting by accurately classifying waste images into organic and recyclable categories.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Deployment](#deployment)
- [Results](#results)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [References](#references)

---

## Introduction

Waste management is a critical environmental challenge, with improper waste sorting leading to pollution and inefficient recycling. This project develops an automated system using deep learning techniques to classify waste images into two categories: **Organic** and **Recyclable**. By leveraging a CNN architecture and data augmentation techniques, the system aims to improve accuracy in waste classification and reduce the reliance on manual sorting.

---

## Features

- **Custom CNN Model:**  
  - Built with multiple convolutional layers, BatchNormalization, and Dropout to improve accuracy and reduce overfitting.
  
- **Data Preprocessing & Augmentation:**  
  - Standardizes images to a uniform size (224x224) and normalizes pixel values.
  - Utilizes augmentation techniques (rotation, shifts, flips, zoom) to enhance model robustness.

- **Optimized Training:**  
  - Uses callbacks such as EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau to optimize training performance.

- **Interactive Web Deployment:**  
  - Deploys a user-friendly Streamlit web app for real-time waste classification.
  - Allows users to upload images and instantly view predictions.

---

## Installation

Ensure you have Python 3.7 or later installed. Then, install the required dependencies:

--pip install tensorflow keras opencv-python-headless pillow matplotlib seaborn tqdm streamlit kagglehub

**Dataset**
The dataset for this project is sourced from Kaggle using the kagglehub package. The dataset includes images organized into two directories:

  - TRAIN: Contains images for training the model, organized by waste category (e.g., Organic, Recyclable).
  - TEST: Contains images for evaluating the model.
The dataset is downloaded automatically via the following code snippet in the project:

    >>>import kagglehub
    >>>path = kagglehub.dataset_download("techsash/waste-classification-data")

**Methodology**
  1. Data Collection & Preparation:
    -Download and organize the dataset.
    -Visualize the class distribution and sample images.
  2. Data Preprocessing & Augmentation:
    -Resize images to 224x224 pixels.
    -Normalize pixel values to [0, 1].
    -Apply augmentation techniques to increase data diversity.
  3. Model Architecture:
    -Build a CNN with convolutional layers, BatchNormalization, activation functions, pooling, dropout, and dense layers.
    -Use softmax activation in the final layer for categorical classification.
  4. Training Process:
    -Compile the model using the Adam optimizer and categorical crossentropy loss.
    -Incorporate callbacks such as EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau for effective training.
  5. Deployment:
    -Develop an interactive Streamlit web app to allow users to upload images and receive real-time waste classification predictions.

**Running the Web App**
  -To deploy the interactive web application:
  -Ensure that the trained model file (e.g., best_model.h5) is available in the project directory.
  -Run the following command:
      >>>streamlit run app.py
**Model Architecture**
  -The CNN model is structured as follows:

  *Convolutional Blocks:*
    -> Block 1: Conv2D (32 filters) → BatchNormalization → ReLU Activation → MaxPooling2D   
    -> Block 2: Conv2D (64 filters) → BatchNormalization → ReLU Activation → MaxPooling2D   
    -> Block 3: Conv2D (128 filters) → BatchNormalization → ReLU Activation → MaxPooling2D   
  *Dense Layers:*
    -> Flatten → Dense (256 units) → BatchNormalization → ReLU Activation → Dropout   
    -> Dense (64 units) → BatchNormalization → ReLU Activation → Dropout    
    -> Final Dense Layer with softmax activation for 2 classes.   

**Training & Evaluation**
  1.Compilation:
    - Optimizer: Adam
    - Loss Function: categorical_crossentropy
    - Metrics: Accuracy
  2.Callbacks:
    - EarlyStopping: Stops training if validation loss doesn’t improve for a set number of epochs.
    - ModelCheckpoint: Saves the best model based on validation performance.
ReduceLROnPlateau: Reduces the learning rate when performance stagnates.
Evaluation Metrics:

Accuracy and loss curves plotted over epochs.
Confusion matrix and classification reports generated for detailed performance analysis.

**Deployment**
   The deployment phase uses Streamlit to create a web interface:

  1.File Upload:
    - Users can upload an image through the web interface.
  2.Real-Time Inference:
    - The uploaded image is preprocessed (resized, normalized) and fed into the trained CNN model.
    - The model predicts whether the image shows organic or recyclable waste.
  3.Display:
    -The app displays the uploaded image along with the prediction result.

  **Results**
  - The developed model effectively classifies waste images into two categories.
  - The use of data augmentation and advanced training techniques led to improved generalization.
  - The interactive Streamlit app successfully demonstrates real-time waste classification, making the system accessible for practical use.


  **Conclusion**
This project demonstrates an end-to-end solution for automated waste classification, addressing the challenges of manual sorting and environmental inefficiencies. By leveraging CNNs, advanced data processing techniques, and an interactive deployment using Streamlit, the system not only achieves high classification accuracy but also provides a scalable solution for sustainable waste management.

- References
TensorFlow Documentation - https://www.tensorflow.org/
Keras Documentation - https://www.tensorflow.org/guide/keras
OpenCV Documentation - https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
Streamlit Documentation - https://docs.streamlit.io/
Waste Classification Dataset on Kaggle - https://www.kaggle.com/datasets/techsash/waste-classification-data/data
