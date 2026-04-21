#  Chili Detection and Classification AI Model

**Author:** Le Thanh Sang  

---

##  Overview  
This project presents an AI model for detecting and classifying chili quality from images. It is designed to support industrial production lines by automatically identifying and filtering out low-quality chili materials.  

The model classifies chili into three categories: **Normal, Dried, and Defective (low-quality).**

---

##  Methodology  
The system follows a two-stage pipeline:  
- **Object Detection:** Chili objects are detected using YOLO.  
- **Classification:** Detected regions are classified using a CNN based on the GoogLeNet architecture.  

By focusing only on detected chili regions instead of full images, the model improves both efficiency and accuracy.

---

##  Application  
- Input quality control in chili production lines  
- Detection of defective chili on conveyor belts  
- Support for raw material selection  

---

##  Advantages  
- Combines YOLO (detection) and GoogLeNet (classification)  
- Improved accuracy by focusing on detected objects  
- Suitable for real-time industrial applications  
- Utilizes a modern deep learning architecture  
<Figure size 640x480 with 3 Axes>
---

##  Limitations  
- Limited training data affects performance  
- Difficulty in accurately distinguishing defective chili in some cases  

---

##  Documentation  
A detailed report is included in this project for further explanation of the model, training process, and results.

---

##  Note  
Please ensure all file paths are correctly updated according to your local environment before running the code.

---

##  Conclusion  
This project demonstrates the effectiveness of combining object detection and deep learning classification for automated quality control. With more training data and further optimization, the model can achieve higher performance in real-world applications.
