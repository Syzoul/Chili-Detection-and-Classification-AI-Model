#Chili Detection and Classification AI Model

Author

Le Thanh Sang

Overview

This project presents an AI model for detecting and classifying chili quality from images. The system is designed to support industrial production lines by automatically identifying and filtering out low-quality chili materials.

The model classifies chili into the following categories:

Normal

Dried

Defective (low-quality)

Methodology

The proposed system follows a two-stage pipeline:

Object Detection:

Chili objects are first detected using the YOLO model.

Classification:

Detected chili regions are then classified using a Convolutional Neural Network based on the GoogLeNet architecture.

Instead of processing the entire image, this approach focuses only on detected chili regions, improving both efficiency and accuracy.

Application

This AI model is designed for industrial automation, specifically:

Input quality control in chili production lines

Detecting defective chili on conveyor belts

Supporting raw material selection processes

Advantages

Combines YOLO (detection) and GoogLeNet (classification)

Improves accuracy by focusing on detected objects

Suitable for real-time applications in production environments

Uses a modern deep learning architecture

<Figure size 640x480 with 3 Axes>

Limitations

Limited training data affects overall performance

The model may not accurately distinguish defective chili in some cases

Documentation

A detailed report is included in this project.

Please refer to it for a deeper explanation of the model, training process, and results.

Conclusion

This project demonstrates the effectiveness of combining object detection and deep learning classification for automated quality control. With more training data and further optimization, the model can achieve even higher performance in real-world applications.

Note

When using this code, please ensure that all file paths are correctly updated to match your local environment.
