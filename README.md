# CIFAR-10 Image Classification

This project implements an image classification system for the CIFAR-10 dataset using a pre-trained ResNet50 model. The system includes a **Flask API** for serving predictions and a **Streamlit frontend** for user interaction. The application is containerized using **Docker** for easy deployment.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Setup Instructions](#setup-instructions)
   - [Prerequisites](#prerequisites)
   - [Local Setup](#local-setup)
4. [Usage](#usage)
   - [Flask API](#flask-api)
   - [Streamlit Frontend](#streamlit-frontend)
5. [Deployment](#deployment)

---

## Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes (e.g., airplane, automobile, bird, etc.). This project uses a pre-trained ResNet50 model to classify images from the dataset. The system includes:

- A **Flask API** for serving predictions.
- A **Streamlit frontend** for uploading images and viewing predictions.
- **Docker** containerization for easy deployment.

---

## Data Preprocessing and Feature Engineering

- **Loading the Dataset**: The CIFAR-10 dataset was loaded using TensorFlow/Keras. It consists of 60,000 32x32 color images in 10 classes (e.g., airplane, automobile, bird, etc.).
- **Exploratory Data Analysis (EDA)**: Visualized sample images from each class to understand the dataset. Checked the distribution of classes to ensure balance.
- **Preprocessing**: Normalization: Pixel values were scaled to the range [0, 1] by dividing by 255. Resizing and One-Hot Encoding was also performed.
- **Data Augmentation**: Applied techniques like rotation, flipping, and zooming to increase the diversity of the training data and reduce overfitting.

## Model Selection and Optimization

- **Model Selection**: ResNet50 was chosen as the base model due to its strong performance on image classification tasks. Two approaches were experimented, One was simple ResNet18 from scratch and the other was Pretrained ResNet50 Architecture trained on ImageNet Weights.
- **Model Architecture**: Added custom layers on top of the pre-trained ResNet50: Global Average Pooling. Fully connected layers with ReLU activation. Dropout layers for regularization. Softmax output layer for multi-class classification.
- **Optimization**: Categorical cross-entropy, Adam Optimizer, Early Stopping and Learning Rate Scheduling was performed to optimize the model.

## Deployment

- **Flask API**: A Flask API was created to serve predictions. The /predict endpoint accepts an image file and returns the predicted class. Basic authentication was implemented for secure access.

- **Streamlit Frontend**: A lightweight frontend was built using Streamlit to allow users to upload images and view predictions.

- **Docker Containerization**: The application was containerized using Docker for easy deployment. Docker Compose was used to manage multiple containers (Flask API and Streamlit app).

- **CLoud API Deployment**: The application was deployed on Render for public access.
  [Render](https://resnet-app.onrender.com)

## Features

- **Image Classification**: Classify images into one of 10 CIFAR-10 classes.
- **Flask API**: RESTful API for serving predictions.
- **Streamlit Frontend**: User-friendly interface for uploading images and viewing predictions.
- **Docker Support**: Containerized application for easy deployment.
- **Basic Authentication**: Secure API access with username and password.

---

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerization)
- Git (optional, for cloning the repository)

### Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/shreyas025/resnet-app.git
   cd resnet-app

2. Install Dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Flask API:
   ```bash
   python app.py

4. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py

---

## Usage

### Flask API

- **Endpoint**: /predict
- **Method**: POST
- **Authentication**: Basic Auth (username: admin, password: password)
- **Input**: Image file
- **Output**: Predicted Class

5. Example using CURL
   ```bash
   curl.exe -X POST -u admin:password -F "file=@path/to/image.jpg" http://localhost:8000/predict
   
### Streamlit Frontend

- Open the streamlit app by running the script streamlit_app.py
- Upload an image using the file uploader.
- View the predicted class.

## Deployment

- The api is deployed on render (free api service). However the inference is unable to produce results as our current tensorflow model cannot load as it exceeds the permisible size.
