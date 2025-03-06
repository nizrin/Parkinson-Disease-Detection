# Parkinson's Disease Detection

# Overview
This project implements a machine learning model to detect Parkinson's disease based on voice measurements. The dataset is preprocessed, and multiple machine learning algorithms are trained to determine the best-performing model.

## Dataset
The dataset used is **parkinsons.data**, which contains various biomedical voice measurements from people with and without Parkinson's disease.

### Features:
- Various voice measurements including fundamental frequency, jitter, shimmer, and noise-to-harmonics ratio.
- **Status**: Target variable (1 = Parkinson's, 0 = Healthy).

## Technologies & Libraries Used
- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib & Seaborn** (for visualization)
- **Scikit-Learn** (for machine learning models)

## Steps Implemented

### 1. Data Loading & Preprocessing
- Load the dataset using Pandas.
- Drop the **name** column as it is not relevant.
- Define features (**X**) and target variable (**y**).
- Split the dataset into training and testing sets (80-20 split).
- Apply **Standardization** and **Normalization** to improve model performance.

### 2. Model Selection & Training
- Implemented the following models:
  - **Support Vector Machine (SVM)**
  - **K-Nearest Neighbors (KNN)**
  - **Decision Tree Classifier**
  - **Random Forest Classifier**
- Evaluated all models based on **accuracy score**.
- Selected the best-performing model (Random Forest in case of a tie with KNN).

### 3. Model Evaluation
- Displayed the **classification report** and **confusion matrix** for the best model.
- Visualized the confusion matrix using Seaborn.

### 4. Building a Predictive System
- Created a function to predict Parkinson's disease based on user input.
- Applied the same scaling and normalization transformations to the input data.
- Provided an example usage of the prediction system.

## Accuracy Results
- **KNN**: 95%
- **Random Forest**: 95%
- **SVM**: 90%
- **Decision Tree**: Lower than others
- **Best Model Selected**: **Random Forest (due to tie-breaking priority)**


## Conclusion
This project successfully implements a **machine learning-based Parkinson’s detection system**. By using feature scaling and selecting the most accurate model, the system can provide high-confidence predictions for early disease detection.

---
### 📌 Future Enhancements
- Experiment with deep learning models.
- Improve feature selection for better accuracy.
- Deploy the model using Flask or Streamlit for a user-friendly interface.


---


