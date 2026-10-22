# 🧠 Deep Learning PR1 — Breast Cancer Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikitlearn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Project-Deep%20Learning-purple" alt="Deep Learning Project">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Project Status">
</p>

## 📌 Project Overview

This project explores **deep learning techniques for breast cancer classification** using the Breast Cancer Wisconsin Diagnostic dataset.

The main objective is to develop, train, evaluate, and compare neural network models that classify breast cancer cases as **malignant or benign** using numerical features extracted from digitized images of breast mass samples.

The project investigates different neural network architectures, activation functions, training strategies, dropout, and regularization techniques to understand their effects on classification performance.

> ⚠️ **Disclaimer:** This project is for educational purposes only. It is not intended to provide medical diagnoses or replace advice from qualified healthcare professionals.

---

## 🎯 Objectives

* 🧠 Build a Single-Layer Perceptron (SLP) as a baseline model.
* 🔬 Develop Multilayer Perceptron (MLP) models.
* ⚡ Compare ReLU, Tanh, and Sigmoid hidden-layer activation functions.
* ⏹️ Evaluate Early Stopping to reduce unnecessary training.
* 🛡️ Investigate dropout at different rates.
* 📉 Compare L1, L2, and combined L1 + L2 regularization.
* 📊 Evaluate model performance using appropriate classification metrics.
* 🔍 Analyze generalization and potential overfitting.
* 📝 Document findings, limitations, and possible improvements.

---

## 📂 Project Structure

```text
DL_PR1/
│
├── DL_PR1.ipynb
├── README.md
├── requirements.txt
└── DL_PR1_Explanation.mp4
```

| File                     | Description                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------- |
| `DL_PR1.ipynb`           | Main Jupyter Notebook containing the code, experiments, results, graphs, and analysis. |
| `README.md`              | Project overview, setup instructions, and results.                                     |
| `requirements.txt`       | Python dependencies required to run the notebook.                                      |
| `DL_PR1_Explanation.mp4` | Project explanation video, if required by the submission instructions.                 |

---

## 📊 Dataset Information

**Dataset:** Breast Cancer Wisconsin Diagnostic (WDBC)

**Source:** UCI Machine Learning Repository

The dataset contains numerical measurements describing characteristics of cell nuclei in breast mass samples.

| Property               | Value |
| ---------------------- | ----- |
| Total samples          | 569   |
| Input features         | 30    |
| Classification classes | 2     |
| Malignant label        | 0     |
| Benign label           | 1     |
| Training samples       | 386   |
| Validation samples     | 69    |
| Test samples           | 114   |

### 🏷️ Classification Labels

* 🔴 **Malignant (0):** Indicates a malignant class in the dataset.
* 🟢 **Benign (1):** Indicates a benign class in the dataset.

The dataset is divided into training, validation, and test sets. Feature scaling is fitted on the training data to avoid using validation or test information during preprocessing.

---

## 🧪 Experiments Performed

### 1️⃣ Single-Layer Perceptron (SLP)

A baseline neural network was developed to establish an initial performance reference.

### 2️⃣ Multilayer Perceptron (MLP)

A multilayer neural network was developed to investigate whether additional hidden layers improve classification performance.

### 3️⃣ Activation Function Comparison

The following hidden-layer activation functions were compared:

* ReLU
* Tanh
* Sigmoid

### 4️⃣ Early Stopping

Early Stopping was evaluated to stop training when validation loss no longer improves and restore the best model weights.

### 5️⃣ Dropout Regularization

Different dropout rates were compared to investigate their effect on generalization and test performance.

### 6️⃣ L1 and L2 Regularization

The following configurations were evaluated:

* No regularization
* L1 regularization
* L2 regularization
* Combined L1 + L2 regularization

### 7️⃣ Final Comparison and Analysis

The recorded results from the experiments were compared. Business insights, limitations, future improvements, and notebook quality checks were documented.

---

## 📈 Recorded Results

The following results are based on the experiments recorded in the notebook.

| Model                        | Validation Accuracy | Test Accuracy |
| ---------------------------- | ------------------: | ------------: |
| SLP Baseline                 |                 N/A |        95.61% |
| MLP — ReLU                   |              97.10% |           N/A |
| MLP — Tanh                   |              98.55% |        97.37% |
| MLP — Sigmoid Hidden Layers  |              97.10% |           N/A |
| MLP — Early Stopping         |              97.10% |        97.37% |
| MLP — Dropout 0.1            |              98.55% |        96.49% |
| MLP — Dropout 0.3            |              98.55% |        96.49% |
| MLP — Dropout 0.5            |              97.10% |        95.61% |
| MLP — No Regularization      |              97.10% |        97.37% |
| MLP — L1 Regularization      |              95.65% |        95.61% |
| MLP — L2 Regularization      |              95.65% |        98.25% |
| MLP — L1 + L2 Regularization |              97.10% |        96.49% |

**Notes:**

* `N/A` means the test result was not available in the recorded experiment summary.
* The highest recorded validation accuracy was **98.55%**.
* The highest recorded test accuracy was **98.25%**, achieved by the L2-regularized model.
* Test accuracy is reported for evaluation and should not be used retrospectively to select a model.

---

## 📏 Evaluation Metrics

The models were evaluated using the following metrics:

* 🎯 **Accuracy:** Overall proportion of correctly classified samples.
* 🔎 **Precision:** Proportion of predicted positive cases that are actually positive for the selected class.
* 🩺 **Recall:** Proportion of actual cases of a class that are correctly identified.
* ⚖️ **F1-Score:** Harmonic mean of precision and recall.
* 📉 **Loss:** Measures the model's prediction error according to its loss function.
* 🔄 **Generalization Gap:** Difference between training and validation performance.

For this classification task, malignant recall is particularly important because incorrectly classifying a malignant case as benign can have serious consequences.

---

## 🛠️ Technologies Used

* 🐍 Python
* 📓 Jupyter Notebook
* 🧠 TensorFlow
* 🔗 Keras
* 📊 NumPy
* 🗃️ Pandas
* 📈 Matplotlib
* 🎨 Seaborn
* 🔬 Scikit-learn

---

## 💻 Installation and Setup

### Prerequisites

Install Python and ensure that `pip` is available.

Python 3.10 or a compatible version is recommended, subject to the compatibility of the installed TensorFlow version.

### 1. Clone the Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/DL_PR1.git
```

### 2. Navigate to the Project Folder

```bash
cd DL_PR1
```

### 3. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `DL_PR1.ipynb` and execute the cells in order.

> 💡 If TensorFlow installation fails, check your Python version and the TensorFlow installation requirements for your operating system.

---

## 📦 Requirements

Create a `requirements.txt` file containing the packages required by the notebook.

For example:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
jupyter
```

For reproducible results, use compatible package versions and record the versions tested in your environment.

---

## 🔍 Key Findings

* 🧠 Multilayer neural networks were evaluated against a single-layer baseline.
* ⚡ Different activation functions produced different validation results.
* ⏹️ Early Stopping helped control training duration and restore the best validation-loss weights.
* 🛡️ Dropout performance varied according to the dropout rate.
* 📉 Regularization did not consistently improve performance across all recorded experiments.
* 🏆 The L2-regularized model achieved the highest recorded test accuracy of 98.25%.
* 🔬 Further evaluation is necessary to establish whether the results generalize to independent datasets.

---

## ⚠️ Limitations

* The dataset contains a limited number of samples.
* The test set contains only 114 samples.
* Results may vary with random seeds, data splits, and model configurations.
* Test metrics for some activation-function experiments were unavailable in the recorded summary.
* High classification accuracy does not establish suitability for real-world medical diagnosis.
* Independent evaluation and appropriate clinical validation would be required before any practical medical use.

---

## 🚀 Future Improvements

* 📚 Evaluate larger and more diverse datasets.
* 🔄 Apply repeated stratified cross-validation.
* 📊 Record consistent test metrics for every model.
* 🔍 Investigate confusion matrices and false-negative predictions.
* ⚙️ Tune hyperparameters using validation data.
* 🧪 Evaluate the final model on an independent dataset.
* 🏥 Conduct appropriate clinical validation before considering real-world applications.

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Building neural networks with TensorFlow and Keras.
* Preparing and scaling a classification dataset.
* Comparing activation functions.
* Applying Early Stopping, dropout, and regularization.
* Evaluating classification performance.
* Interpreting model results and generalization.
* Organizing and documenting a machine learning project.
* Understanding the importance of responsible AI in healthcare-related applications.

---

## 👨‍💻 Author

**Name:** Your Name

**University:** Your University Name

**Project:** Deep Learning PR1

**Domain:** Deep Learning | Machine Learning | Classification

🔗 **GitHub:** https://github.com/YOUR_USERNAME

---

## 📜 Disclaimer

This project is an educational machine learning experiment using a publicly available dataset. It is not a certified medical tool and must not be used independently for diagnosis, treatment, or clinical decision-making.

---

⭐ If you find this project interesting, feel free to explore the notebook and its experiments!

