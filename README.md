### Traffic Sign Classifier - README

#### Overview
This repository contains a **Traffic Sign Classification System** developed using deep learning. The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset, accessed via the DeepLake library, to train a Convolutional Neural Network (CNN). The classifier predicts 43 different traffic sign classes, demonstrating the potential of machine learning in intelligent traffic systems and autonomous vehicles.

---

#### Features
- **Dataset Access via DeepLake:** Simplifies dataset management and ensures efficient loading.
- **Data Preprocessing:** Handles image normalization and label encoding.
- **Model Training:** Implements a CNN architecture optimized for traffic sign recognition.
- **Performance Metrics:** Evaluates the model using accuracy, precision, recall, and F1-score.
- **Final Test Accuracy:** Achieves an impressive **76.77% accuracy** on the test dataset.

---

#### Dataset
The dataset used is the **German Traffic Sign Recognition Benchmark (GTSRB)**.

**Source:**  
The GTSRB dataset is available on [Link](https://datasets.activeloop.ai/docs/ml/datasets/gtsrb-dataset/). It has been imported into this project using the DeepLake library for streamlined data handling.

**Steps to Import the Dataset with DeepLake:**
1. Install the DeepLake library:
   ```bash
   pip install deeplake
   ```
2. Load the dataset:
   ```python
   import deeplake

   # Load training and testing datasets
   train = deeplake.load("path/to/train/deeplake-dataset")
   test = deeplake.load("path/to/test/deeplake-dataset")
   ```

DeepLake provides efficient access to large-scale datasets, ensuring seamless integration into machine learning pipelines.

---

#### Project Details
**Preprocessing:**  
- **Images:** Normalized to scale pixel values between [0, 1].  
- **Labels:** One-hot encoded for multi-class classification.

**Model Architecture:**  
The CNN model is built with multiple convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification. Dropout regularization is used to mitigate overfitting.

**Training:**  
The model uses categorical cross-entropy as the loss function and the Adam optimizer for weight updates. Training progresses over multiple epochs, ensuring convergence to optimal performance.

**Evaluation:**  
Class-wise metrics, including precision, recall, and F1-score, provide insights into the model's performance for each traffic sign category.

**Final Outcome:**  
- **Test Accuracy:** 76.77%  
- **Class-wise Metrics:** The model performs exceptionally well for most classes, with some challenges in categories with low inter-class variance or imbalanced data distribution. Despite these, the overall results demonstrate the system's robustness and effectiveness.

**Justification of Results:**  
The achieved accuracy of 76.77% is commendable for a 43-class classification problem, especially when addressing a real-world dataset with varying lighting, angles, and resolution. Misclassifications highlight areas for improvement, such as augmenting the dataset or refining the CNN architecture. The results are promising for deployment in intelligent traffic systems.

---

#### How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/traffic-sign-classifier.git
   cd traffic-sign-classifier
   ```
2. Execute the Jupyter Notebook:
   ```bash
   jupyter notebook traffic_sign_classifier.ipynb
   ```
3. Follow the steps in the notebook to preprocess the data, train the model, and evaluate its performance.

---

#### Conclusion
This project demonstrates the effectiveness of deep learning models in recognizing traffic signs, laying the groundwork for applications in autonomous vehicles and smart traffic systems. The use of DeepLake enhances dataset management and facilitates scalable solutions.

---

#### Contributions
You are welcome to fork this repository, report issues, or create pull requests for further development.

---
