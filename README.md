# Automatic Number Plate Recognition System
This project implements an Automatic Number Plate Recognition (ANPR) system using _YOLOv10_ and OpenCV. 
The system detects vehicle license plates in images or video streams and recognizes the alphanumeric characters on them. 
Such systems are instrumental in traffic monitoring, toll collection, and security enforcement.

📌 Features
Real-time license plate detection using Haar Cascade classifiers.

Character segmentation and recognition using OpenCV.

Outputs processed video with annotated license plates.

⚠️ Limitations
- The current Haar Cascade classifier may not detect all license plate formats, especially non-standard or obscured plates.

- Recognition accuracy may decrease under poor lighting or low-resolution conditions.

- The system primarily supports license plates similar to the training data of the Haar Cascade used.

- DATASET USED IN THIS PROJECT: https://www.kaggle.com/datasets/vikaschauhan734/vehicle-number-plate

- **Sample validation images**: 
- ![val_batch0_labels](https://github.com/user-attachments/assets/8273013f-5049-48a4-b8c6-9953be57af1b)
- ![val_batch0_pred](https://github.com/user-attachments/assets/49afd68c-dea2-42d7-9272-343f554e3994)
- ![val_batch1_pred](https://github.com/user-attachments/assets/22411d34-122c-499e-a83c-a9d6a6e4f9a8)
- **In the _'License-Plate-Detector'_ folder You can see the full validation images,training image sets and other factors like reselt.csv, confution matrix**






