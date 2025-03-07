Desertification Analysis and Image Enhancement using CNN, EESRGAN, and CycleGAN
Project Overview
This project aims to analyze desertification patterns using deep learning techniques, including Convolutional Neural Networks (CNN), Enhanced ESRGAN (EESRGAN), and CycleGAN. The model processes satellite images to detect changes in vegetation, apply super-resolution techniques, and generate high-quality enhanced images for better analysis.

Features
CNN Classification: Identifies desertification regions from satellite images.
Fast Fourier Transform (FFT): Extracts frequency-based features from images for analysis.
EESRGAN (Enhanced Super-Resolution GAN): Enhances image quality using an advanced deep learning model.
CycleGAN: Generates high-quality desertification transformation images from raw satellite images.
NDVI Calculation: Computes the Normalized Difference Vegetation Index (NDVI) for vegetation monitoring.
Performance Metrics: Evaluates SSIM, PSNR, and MSE to assess model performance.
Dataset
The dataset consists of satellite images stored in Google Drive and loaded into the model using Google Colab. Images are resized to 128x128 pixels and normalized for processing.

Installation & Requirements
Run the following to install the necessary dependencies:
pip install numpy opencv-python tensorflow keras scikit-image scikit-learn matplotlib plotly
Usage
Load Dataset:
Place images in the desert folder inside Google Drive.
Mount Google Drive in Google Colab.
Preprocessing:
Apply FFT to extract frequency features.
Compute NDVI for vegetation monitoring.
Train CNN for Classification:
Trained with sparse categorical cross-entropy loss.
Train EESRGAN for Super-Resolution:
Improves image quality using a deep learning-based upscaling approach.
Apply CycleGAN for Image-to-Image Translation:
Enhances desertification images to generate realistic transformations.
Model Performance & Evaluation
CNN Classification Accuracy: Evaluated using metrics like Accuracy, Precision, Recall, and F1-score.
EESRGAN & CycleGAN Image Quality: Assessed using SSIM, PSNR, and MSE.
Future Scope
Integrate time-series analysis to track desertification trends over years.
Use GAN-based domain adaptation for multi-region applicability.
Improve real-time satellite image enhancement for remote sensing applications.
Contributors
Rishabh Jain
R.V. College of Engineering 
E-mail- Rishabhjain1922@gmail.com
