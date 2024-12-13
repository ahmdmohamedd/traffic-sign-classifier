### Traffic Sign Classifier - README

#### Overview
This repository contains a **Traffic Sign Classification System** developed using deep learning. The project utilizes the German Traffic Sign Recognition Benchmark (GTSRB) dataset to train a Convolutional Neural Network (CNN) for classifying traffic signs into 43 distinct categories. This system serves as a stepping stone for intelligent traffic systems and autonomous vehicles, highlighting the potential of machine learning in real-world applications.

---

#### Features
- **Data Preprocessing:** Efficiently preprocesses the dataset for training and evaluation.
- **Model Training:** Implements a CNN architecture tailored to traffic sign recognition.
- **Performance Metrics:** Evaluates the model using accuracy, precision, recall, and F1-score.
- **Final Test Accuracy:** Achieves a commendable **76.77% accuracy** on the test dataset.

---

#### Dataset
The dataset used in this project is the **German Traffic Sign Recognition Benchmark (GTSRB)**. It contains over 50,000 images of 43 different traffic sign classes.

**Source:**  
The dataset can be downloaded from the following link:  
[http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset](https://datasets.activeloop.ai/docs/ml/datasets/gtsrb-dataset/)



**Steps to Import the Dataset:**
1. Download the dataset from the official source.
2. Extract the contents of the dataset.
3. Structure the data into `train` and `test` directories containing respective images and labels.
4. Use the following Python code snippet to load the dataset into your project:
   ```python
   import tensorflow as tf
   
   # Define the file paths
   train_data_dir = "path/to/train_directory"
   test_data_dir = "path/to/test_directory"

   # Load the dataset
   train = tf.keras.preprocessing.image_dataset_from_directory(
       train_data_dir,
       image_size=(32, 32),
       label_mode='int'
   )

   test = tf.keras.preprocessing.image_dataset_from_directory(
       test_data_dir,
       image_size=(32, 32),
       label_mode='int'
   )
   ```

---

#### Project Details
**Preprocessing:**  
Images are normalized to have pixel values in the range [0, 1]. Labels are one-hot encoded for multi-class classification.

**Model Architecture:**  
A CNN is designed with multiple convolutional and pooling layers followed by fully connected layers. This architecture ensures feature extraction and accurate classification of traffic signs.

**Training:**  
The model is trained using the categorical cross-entropy loss function and Adam optimizer. Regularization techniques like dropout are employed to reduce overfitting.

**Evaluation:**  
The model's performance is evaluated on unseen test data, reporting key metrics like precision, recall, and F1-score for each class.

**Final Outcome:**  
- **Test Accuracy:** 76.77%
- **Precision, Recall, and F1-score:** Detailed class-wise metrics are included in the output, showcasing the model's ability to distinguish traffic signs effectively. Despite limitations for certain classes, the overall results demonstrate the system's robustness.

**Justification of Results:**  
Achieving 76.77% accuracy in a multi-class classification task with 43 classes highlights the effectiveness of the CNN model. While certain classes, such as rare or visually similar signs, pose challenges due to class imbalance and low inter-class variance, the overall performance underscores the system's applicability in real-world scenarios. Further improvements could include augmenting the dataset and enhancing the model's architecture.

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
This project demonstrates the potential of deep learning in traffic sign recognition, offering a solid foundation for further advancements. The results achieved underscore the feasibility of applying machine learning in intelligent traffic management and autonomous driving systems.

---

#### Contributions
Feel free to fork this repository, open issues, or submit pull requests to contribute to its development.

---
