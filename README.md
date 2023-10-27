**Eye Cancer Image Classifier**

This project aims to revolutionize the early detection of eye cancer using artificial intelligence and deep learning. Eye cancer, though rare, can have devastating consequences, and early detection is crucial for timely intervention. Our team has developed an AI-driven image classification system to assist medical professionals and empower individuals to take control of their ocular health.

**Table of Contents:**

- [Introduction](#introduction)
- [Problem Analysis](#problem-analysis)
- [Solution Design](#solution-design)
- [Technology](#technology)
- [How to Download the Model](#how-to-download-the-model)

## Introduction

In a world where artificial intelligence and deep learning are reshaping healthcare, our project focuses on the early detection of eye cancer. We have developed an advanced AI-powered image classifier to identify signs of eye cancer in X-ray images. Our goal is to enhance diagnostic capabilities, assist medical professionals, and make eye cancer detection more accessible.

## Problem Analysis

The "Eye Cancer Image Classifier" project involves various challenges:

- Achieving high medical accuracy and reliability.
- Ensuring the quality of the dataset and addressing biases.
- Adhering to ethical and regulatory compliance.
- Building interpretable AI models.
- Creating a user-friendly interface.
- Ensuring scalability and reliability during model deployment.
- Validation against external datasets and involving medical experts.
- Effective communication with stakeholders.

## Solution Design

Our solution comprises the following components:

**User-Centric Platform**
- A user-friendly interface designed for medical professionals.
- Intuitive and powerful tools for efficient diagnosis.

**The Role of Backpropagation**
- Backpropagation capabilities to explain the model's decisions.

**AI Models**
- Two AI models: one for accuracy and one for interpretability.
- The primary image classifier maximizes accuracy, while the second model provides insights into the decision-making process.

**Impact on the Environment**
- Mitigating environmental impact through energy-efficient computing.

## Technology

**Models**
- Custom TensorFlow Model
- Fine-Tuning of HuggingFace Model
- Custom PyTorch Model
- AutoTrain Model

**Server**
- Hosting on Azure Virtual Machines (VMs)

**Platform**
- Web User Interface using React
- Styling with Tailwind CSS

## How to Download the Model

You can download our AI models from the following links:

1. [Interpretability Model (PyTorch Model)](https://drive.google.com/file/d/1bwfiVbSNBQVwIJWINBVcgkv81QB9R2hX/view?usp=sharing)

Download this model and put it in the `models` folder. Then you can run the following command to use the model:

```bash
python3 cli.py test
```

These models are designed to assist medical professionals in the early detection of eye cancer. Please refer to the model-specific documentation for instructions on usage.

We are committed to advancing early disease detection in healthcare and providing tools for medical practitioners to make informed decisions. Feel free to explore and utilize our AI models to contribute to the early detection and treatment of eye cancer.