# Heart Disease Classification - CNN

The main objective of this project is to design and implement an automatic classification method using Convolutional Neural Networks (CNNs) to classify cardiovascular diseases (CVDs). Diseases covered include **Mitral Stenosis (MS)**, **Aortic Stenosis (AS)**, **Mitral Regurgitation (MR)**, **Mitral Valve Prolapse (MVP)**, along with **normal heart sounds**.

## Table of Contents
1. [Background and Objectives](#background-and-objectives)
2. [Methods](#methods)
   - 2.1 [Data Collection and Preprocessing](#data-collection-and-preprocessing)
   - 2.2 [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
   - 2.3 [Multi-Classification of Cardiac Diseases](#multi-classification-of-cardiac-diseases)
   - 2.4 [Training and Evaluation](#training-and-evaluation)
3. [Expected Outcomes](#expected-outcomes)
4. [Installation](#installation)
5. [Usage](#usage)

---

## Background and Objectives

Cardiovascular diseases (CVDs) are a leading cause of death and disability worldwide. Timely and accurate diagnosis is essential to mitigate the impact of these diseases. However, in many remote areas, access to medical professionals and diagnostic equipment is limited, making early detection of these conditions difficult.

**Phonocardiograms (PCG)**, which are recordings of heart sounds, serve as a non-invasive method to assess heart health. In remote areas, where healthcare professionals may not be readily available, automated systems powered by **Artificial Intelligence (AI)** can provide critical support in diagnosing these diseases.

### Objectives:
The primary goal of this research is to design and implement an automated classification system using machine learning techniques, specifically **Convolutional Neural Networks (CNN)**, to classify cardiac diseases from PCG signals. This system should:
1. **Automatically classify multiple cardiac diseases**: Using heart sound recordings to identify and classify diseases like Mitral Stenosis (MS), Aortic Stenosis (AS), Mitral Regurgitation (MR), Mitral Valve Prolapse (MVP), and Normal heart sounds.
2. **Enhance diagnostic accuracy and robustness**: Handle noisy environments effectively through data augmentation, ensuring the model performs well in real-world scenarios.
3. **Provide a solution for resource-limited areas**: Enable faster diagnosis without requiring direct intervention from medical professionals, particularly in underserved regions.

---

## Methods

### 2.1 Data Collection and Preprocessing
- **Phonocardiogram Signals**: The dataset includes heart sound recordings for both normal and abnormal heart conditions.
- **Data Augmentation**: To make the model more robust and handle various environmental factors, data augmentation techniques such as noise injection, pitch shifts, and time shifts are applied.

### 2.2 Convolutional Neural Network (CNN)
- **Model Architecture**: The proposed CNN model contains several convolutional layers designed to learn hierarchical features from the input heart sounds.
- **Activation Functions**: ReLU is used in hidden layers, while softmax is applied in the output layer to classify the sounds into one of five categories.
- **Loss Function & Optimizer**: The model uses categorical cross-entropy for multi-class classification, optimized with the Adam optimizer.

### 2.3 Multi-Classification of Cardiac Diseases
The model is designed to classify heart diseases into the following five categories:
- **Mitral Stenosis (MS)**
- **Aortic Stenosis (AS)**
- **Mitral Regurgitation (MR)**
- **Mitral Valve Prolapse (MVP)**
- **Normal (N)**

Each class is represented by a neuron in the output layer, with the model outputting the class with the highest probability.

### 2.4 Training and Evaluation
- **Training Set**: The model is trained on a large set of labeled PCG signals.
- **Validation and Testing**: The model is validated and tested on separate sets to ensure its generalization capability.
- **Evaluation Metrics**: The model’s performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrix. Additionally, the ROC curve is analyzed to check the model's performance for each class.

---

## Expected Outcomes

1. **Improved Diagnostic Accuracy**: The model aims to achieve high classification accuracy, even in noisy environments, ensuring reliable predictions for different heart conditions.
2. **Efficient Solution for Remote Areas**: The system provides a low-cost, easy-to-use solution that can be deployed in areas where healthcare professionals are scarce.
3. **Model Usability**: The trained model can be integrated into mobile health apps, telemedicine systems, or even deployed locally in healthcare settings to provide real-time diagnostics.
4. **Contribution to AI in Healthcare**: This research adds to the growing application of machine learning in healthcare, specifically for early detection of heart diseases from heart sound signals.

---
## Streamlit Demo

<iframe width="560" height="315" src="https://www.youtube.com/embed/ix7rlCNMh88" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

