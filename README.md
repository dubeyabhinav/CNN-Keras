# Digit Recognizer: CNN with Keras

This repository contains a beginner-friendly guide and implementation of a Convolutional Neural Network (CNN) using Keras to solve the classic Digit Recognizer problem (MNIST dataset). The project walks through the end-to-end process of building a deep learning model, from data preprocessing to model evaluation and prediction.

## 📋 Project Overview

The goal of this project is to correctly identify digits (0-9) from a dataset of tens of thousands of handwritten images. We achieve this by building and training a Convolutional Neural Network (CNN), a class of deep neural networks most commonly applied to analyzing visual imagery.

**Key Features:**
* **Deep Learning Framework:** Keras (running on top of TensorFlow).
* **Model Architecture:** A custom Sequential CNN with Convolutional, Max Pooling, Dropout, and Dense layers.
* **Optimization:** Uses the RMSprop optimizer and a learning rate reduction callback (`ReduceLROnPlateau`).
* **Data Augmentation:** Implements `ImageDataGenerator` to improve model generalization and prevent overfitting.

---

## 📊 1. The Data

The dataset used is the famous MNIST dataset (modified for the Kaggle Digit Recognizer competition).

* **Source:** [Kaggle Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer/data)
* **Input Dimensions:** 28x28 pixels (Grayscale).
* **Content:**
    * **Train Data:** 42,000 labeled images.
    * **Test Data:** 28,000 unlabeled images.

### Data Analysis & Preprocessing
Before feeding the data into the model, several preprocessing steps were taken to ensure optimal performance:

1.  **Normalization:** Pixel values were scaled from the range `[0, 255]` to `[0, 1]` to reduce illumination differences and speed up training convergence.
2.  **Reshaping:** The flattened 1D rows (784 pixels) were reshaped into 3D matrices (`28x28x1`) to represent height, width, and color channels.
3.  **One-Hot Encoding:** The target labels (0-9) were encoded into binary class matrices (e.g., `2` becomes `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]`).
4.  **Train/Val Split:** The training data was split into a training set (90%) and a validation set (10%) to monitor overfitting during training.

![Sample Digits](<img width="422" height="417" alt="digit" src="https://github.com/user-attachments/assets/96f6ad4a-1893-46ae-9879-647301c19391" />
)

---

## 🛠️ 2. The Process

### Model Architecture (CNN)
We constructed a Sequential model using the Keras API with the following structure:

* **Feature Extraction:**
    * 2x Convolutional Layers (32 filters, Kernel 5x5, ReLU)
    * Max Pooling (2x2) + Dropout (0.25)
    * 2x Convolutional Layers (64 filters, Kernel 3x3, ReLU)
    * Max Pooling (2x2) + Dropout (0.25)
* **Classification:**
    * Flatten Layer
    * Dense Layer (256 units, ReLU)
    * Dropout (0.5)
    * Output Dense Layer (10 units, Softmax)

### Training Strategy
* **Optimizer:** `RMSprop` (adjusted with specific rho and epsilon values).
* **Loss Function:** `categorical_crossentropy` (standard for multi-class classification).
* **Callback:** `ReduceLROnPlateau` - This callback dynamically reduces the learning rate by half if the validation accuracy does not improve after 3 epochs.

### Data Augmentation
To prevent overfitting and make the model robust to variations in handwriting, we artificially expanded the training dataset using `ImageDataGenerator`.
* **Techniques applied:** Random rotation (10°), Zoom (10%), Width/Height shifts (10%).
* *Note: Vertical/Horizontal flips were disabled as they can confuse digit orientation (e.g., 6 vs 9).*

---

## 📈 3. Outcomes & Evaluation

### Training Performance
The model was trained for 5 epochs for demonstration purposes, achieving:
* **Training Accuracy:** ~97%
* **Validation Accuracy:** ~98-99%

*(Note: Extending training to 20+ epochs typically pushes accuracy to >99%)*

![Learning Curves](<img width="562" height="465" alt="acc" src="https://github.com/user-attachments/assets/656553af-d09a-4afe-939c-0571f93092ea" />
)

### Confusion Matrix
To evaluate specific weaknesses, we plotted a confusion matrix on the validation set. This highlights which digits are most frequently confused (e.g., distinguishing between a '4' and a '9').

![Confusion Matrix](<img width="549" height="487" alt="cfm" src="https://github.com/user-attachments/assets/fef1a359-33d8-4e15-8184-bf0404508abc" />
)

### Error Analysis
We inspected the most significant errors by calculating the difference between the predicted probability and the true label. The top errors often reveal digits that are poorly written or ambiguous even to the human eye.

![Top Errors](<img width="548" height="423" alt="res" src="https://github.com/user-attachments/assets/0b2ccdac-ca13-4122-adff-a48c2d1012fb" />
)

---

## 💻 Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/digit-recognizer-cnn.git](https://github.com/your-username/digit-recognizer-cnn.git)
    cd digit-recognizer-cnn
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
    ```

3.  **Run the notebook:**
    Open `cnn-keras-guide-0-98.ipynb` in Jupyter Notebook or Google Colab and run the cells sequentially to train the model and generate the `submission.csv`.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
