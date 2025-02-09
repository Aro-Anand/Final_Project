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
- [License](#license)

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

```bash
pip install tensorflow keras opencv-python-headless pillow matplotlib seaborn tqdm streamlit kagglehub
Dataset
The dataset for this project is sourced from Kaggle using the kagglehub package. The dataset includes images organized into two directories:

TRAIN: Contains images for training the model, organized by waste category (e.g., Organic, Recyclable).
TEST: Contains images for evaluating the model.
The dataset is downloaded automatically via the following code snippet in the project:

python
Copy
Edit
import kagglehub
path = kagglehub.dataset_download("techsash/waste-classification-data")
Methodology
Data Collection & Preparation:

Download and organize the dataset.
Visualize the class distribution and sample images.
Data Preprocessing & Augmentation:

Resize images to 224x224 pixels.
Normalize pixel values to [0, 1].
Apply augmentation techniques to increase data diversity.
Model Architecture:

Build a CNN with convolutional layers, BatchNormalization, activation functions, pooling, dropout, and dense layers.
Use softmax activation in the final layer for categorical classification.
Training Process:

Compile the model using the Adam optimizer and categorical crossentropy loss.
Incorporate callbacks such as EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau for effective training.
Deployment:

Develop an interactive Streamlit web app to allow users to upload images and receive real-time waste classification predictions.
Usage
Running the Training Script
To train the model, execute the notebook or script containing the training code. Ensure that your dataset paths are correctly set:

python
Copy
Edit
train_path = "path/to/your/dataset/TRAIN"
test_path = "path/to/your/dataset/TEST"
Then run:

bash
Copy
Edit
python train_model.py
or open the Jupyter Notebook and run all cells sequentially.

Running the Web App
To deploy the interactive web application:

Ensure that the trained model file (e.g., best_model.h5) is available in the project directory.
Run the following command:
bash
Copy
Edit
streamlit run app.py
Open the provided local URL in your browser to interact with the app.
Model Architecture
The CNN model is structured as follows:

Convolutional Blocks:

Block 1: Conv2D (32 filters) → BatchNormalization → ReLU Activation → MaxPooling2D
Block 2: Conv2D (64 filters) → BatchNormalization → ReLU Activation → MaxPooling2D
Block 3: Conv2D (128 filters) → BatchNormalization → ReLU Activation → MaxPooling2D
Dense Layers:

Flatten → Dense (256 units) → BatchNormalization → ReLU Activation → Dropout
Dense (64 units) → BatchNormalization → ReLU Activation → Dropout
Final Dense Layer with softmax activation for 2 classes.
Training & Evaluation
Compilation:

Optimizer: Adam
Loss Function: categorical_crossentropy
Metrics: Accuracy
Callbacks:

EarlyStopping: Stops training if validation loss doesn’t improve for a set number of epochs.
ModelCheckpoint: Saves the best model based on validation performance.
ReduceLROnPlateau: Reduces the learning rate when performance stagnates.
Evaluation Metrics:

Accuracy and loss curves plotted over epochs.
Confusion matrix and classification reports generated for detailed performance analysis.
Deployment
The deployment phase uses Streamlit to create a web interface:

File Upload:
Users can upload an image through the web interface.
Real-Time Inference:
The uploaded image is preprocessed (resized, normalized) and fed into the trained CNN model.
The model predicts whether the image shows organic or recyclable waste.
Display:
The app displays the uploaded image along with the prediction result.
Results
The developed model effectively classifies waste images into two categories.
The use of data augmentation and advanced training techniques led to improved generalization.
The interactive Streamlit app successfully demonstrates real-time waste classification, making the system accessible for practical use.
Future Work
Model Enhancements:

Explore advanced architectures such as transfer learning with MobileNet or ResNet.
Experiment with additional hyperparameter tuning.
Extended Classification:

Expand the project to classify additional waste categories.
Deployment Improvements:

Integrate with cloud services and CI/CD pipelines for scalable deployment.
Enhance the user interface for broader accessibility.
Conclusion
This project demonstrates an end-to-end solution for automated waste classification, addressing the challenges of manual sorting and environmental inefficiencies. By leveraging CNNs, advanced data processing techniques, and an interactive deployment using Streamlit, the system not only achieves high classification accuracy but also provides a scalable solution for sustainable waste management.

References
TensorFlow Documentation
Keras Documentation
OpenCV Documentation
Streamlit Documentation
Waste Classification Dataset on Kaggle
