# Chest X-Ray Pneumonia Classifier

A Deep Learning-based web application for detecting **Pneumonia** from Chest X-Ray images using a Convolutional Neural Network (CNN). The application is built with **TensorFlow/Keras** and deployed using **Streamlit** for an interactive user experience.

---

## 📌 Project Overview

Pneumonia is a serious lung infection that can be identified through Chest X-Ray images. This project uses a Convolutional Neural Network (CNN) to automatically classify X-Ray images into:

* **NORMAL**
* **PNEUMONIA**

The trained model is integrated into a Streamlit web application that allows users to upload X-Ray images and receive instant predictions.

---

## Features

* Upload Chest X-Ray images
* Automatic image preprocessing
* Deep Learning based prediction
* Binary classification:

  * Normal
  * Pneumonia
* Confidence score display
* Interactive Streamlit interface
* Fast and easy deployment

---

## Model Architecture

The CNN model consists of:

* Convolution Layers
* ReLU Activation
* Max Pooling Layers
* Dropout Layers
* Flatten Layer
* Fully Connected Dense Layers
* Sigmoid Output Layer

### Architecture Summary

```text
Input Image
      ↓
Conv2D (64)
      ↓
ReLU
      ↓
MaxPooling
      ↓
Dropout

      ↓

Conv2D (128)
      ↓
ReLU
      ↓
MaxPooling
      ↓
Dropout

      ↓

Conv2D (256)
      ↓
ReLU
      ↓
MaxPooling
      ↓
Dropout

      ↓

Flatten
      ↓

Dense (64)
      ↓

Dropout

      ↓

Dense (1)
      ↓

Sigmoid
```

---

## 📂 Dataset

Dataset used:

**Chest X-Ray Images (Pneumonia)**

Classes:

* NORMAL
* PNEUMONIA

Images were resized and normalized before training.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* OpenCV
* Matplotlib
* Streamlit
* PIL

---

## Project Structure

```text
Chest-XRay-Pneumonia-Classifier/
│
├── app.py
├── custom_pre_trained_model_10.h5 (run the ipnby file and download the pre trained model)
├── requirements.txt
├── README.md
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

download the dataset and add it to this folder

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Chest-XRay-Pneumonia-Classifier.git

cd Chest-XRay-Pneumonia-Classifier
```

### Create Environment

```bash
conda create -n xray python=3.11

conda activate xray
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install tensorflow streamlit pillow numpy pandas opencv-python matplotlib
```

---

## Running the Application

```bash
streamlit run app.py
```

After running the command, open:

```text
http://localhost:8501
```

in your browser.

---

## Model Performance

Evaluation Metrics:

* Accuracy
* Binary Cross Entropy Loss

The model was trained using:

* Adam Optimizer
* Binary Crossentropy Loss
* Batch Size = 32
* Epochs = 10

---

## Workflow

```text
Chest X-Ray Image
        ↓
Image Preprocessing
        ↓
CNN Feature Extraction
        ↓
Classification Layer
        ↓
Prediction
        ↓
NORMAL / PNEUMONIA
```

---

## Application Preview

1. Upload Chest X-Ray Image
2. Click Predict
3. View Prediction Result
4. View Confidence Score

---


## When we run the application-
<img width="1469" height="723" alt="image" src="https://github.com/user-attachments/assets/6d93481a-75d7-4f38-b90b-c5590cf7a676" />

## After prediction-
<img width="1470" height="800" alt="image" src="https://github.com/user-attachments/assets/241bf8e3-c9d1-4080-aabc-b0e464574f50" />
