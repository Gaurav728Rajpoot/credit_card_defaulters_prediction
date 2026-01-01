# Credit Card Default Prediction 💳

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Keras Tuner](https://img.shields.io/badge/Keras%20Tuner-Optimization-red)

## 📌 Project Overview
This project leverages Deep Learning to predict whether a credit card client will default on their payment next month. The goal is to help financial institutions mitigate risk by identifying high-risk clients early.

**Key Achievement:** Transformed a model with high accuracy but poor detection capabilities into a robust risk-detector, **increasing the Recall rate by ~22%** using Class Weighting strategies to handle severe data imbalance.

## ⚠️ The Challenge: The "Accuracy Paradox"
The dataset (UCI Credit Card Data) is highly imbalanced, with ~78% of customers being "Good Payers" and only ~22% being "Defaulters."

* **Initial Baseline:** My first model achieved **82% Accuracy**.
* **The Problem:** Upon inspection, the model was simply guessing "No Default" for almost everyone. It missed **61%** of the actual defaulters (Recall = 0.39), making the model useless for risk management.

## 🛠️ The Solution
I engineered a solution focusing on **Business Value** rather than vanity metrics:

1.  **Deep Learning Architecture:** Built a Sequential Neural Network using Keras with Dropout and Batch Normalization to ensure stability.
2.  **Hyperparameter Tuning:** utilized **Keras Tuner (RandomSearch)** to automate the selection of optimal hyperparameters:
    * Number of units in dense layers (32-512)
    * Learning Rates (0.01, 0.001, 0.0001)
    * Boolean check for Batch Normalization
3.  **Handling Imbalance:** Applied **Class Weights** (`class_weight='balanced'`) to heavily penalize the model for missing a defaulter.
4.  **Metric Optimization:** Shifted focus from Accuracy to **F1-Score** and **Recall**.

## 📊 Results & Impact

| Metric | Baseline Model | Final Model (Weighted) | Change |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 82% | 80% | -2% (Expected Trade-off) |
| **Recall (Defaulters)**| 39% | **61%** | **+22% (Huge Win)** |
| **F1-Score** | 0.48 | 0.53 | +5% |

**Business Interpretation:**
By sacrificing a small amount of total accuracy, the model now captures **22% more risky customers** who would have otherwise caused financial loss. In a real-world banking scenario, detecting these defaults saves significantly more money than the operational cost of a few false alarms.

## 💻 Tech Stack
* **Core:** Python, Pandas, NumPy
* **Machine Learning:** TensorFlow, Keras, Scikit-Learn
* **Optimization:** Keras Tuner
* **Data Vis:** Matplotlib (for training curves)

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/credit-card-default-prediction.git](https://github.com/YOUR_USERNAME/credit-card-default-prediction.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas scikit-learn keras-tuner matplotlib
    ```
3.  **Run the Notebook:**
    Open `Credit_Card_Churn.ipynb` in Jupyter Notebook or Google Colab and run all cells.

## 📈 Future Improvements
* Experiment with SMOTE (Synthetic Minority Over-sampling Technique) to see if it outperforms Class Weights.
* Deploy the model as a simple API using Flask or Streamlit.
