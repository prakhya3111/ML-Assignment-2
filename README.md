# Obesity Level Classification

## 1. Problem Statement

The objective of this project is to build a multi-class classification system to predict an
individual's obesity level based on lifestyle habits, physical attributes, and eating patterns.

The goal is to:
- Implement multiple classification algorithms
- Compare their performance using evaluation metrics
- Deploy the best-performing models using a Streamlit web application

This is a multi-class classification problem with 7 target classes:
- Insufficient_Weight
- Normal_Weight
- Overweight_Level_I
- Overweight_Level_II
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III

## 2. Dataset Description

The dataset used is the **Obesity Levels Dataset**, which contains information about:

- Gender
- Age
- Height
- Weight
- Family history of overweight
- Eating habits (FAVC, FCVC, NCP, CAEC)
- Smoking habits
- Water consumption (CH2O)
- Physical activity (FAF)
- Screen time (TUE)
- Transportation type (MTRANS)

Minimum Requirements:
- Number of features ≥ 12
- Number of instances ≥ 500

The dataset was split into:
- 80% Training data
- 20% Testing data (used for evaluation and Streamlit upload)

## 3. Models Implemented

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

All models were evaluated using the following 6 metrics:

1. Accuracy  
2. AUC Score  
3. Precision  
4. Recall  
5. F1 Score 
6. Matthews Correlation Coefficient (MCC)  

## 4. Model Comparison Table

| ML Model Name        | Accuracy | AUC   | Precision | Recall | F1 Score | MCC    |
|----------------------|----------|-------|-----------|--------|----------|--------|
| Logistic Regression  | 0.8723   | 0.9871| 0.8693    | 0.8688 | 0.8671   | 0.8515 |
| Decision Tree        | 0.9173   | 0.9507| 0.9190    | 0.9151 | 0.9163   | 0.9036 |
| K-Nearest Neighbors  | 0.8274   | 0.9602| 0.8219    | 0.8215 | 0.8110   | 0.8012 |
| Naive Bayes          | 0.5981   | 0.9014| 0.6499    | 0.5945 | 0.5716   | 0.5435 |
| Random Forest        | 0.9527   | 0.9973| 0.9554    | 0.9512 | 0.9521   | 0.9452 |
| XGBoost              | 0.9551   | 0.9976| 0.9577    | 0.9533 | 0.9541   | 0.9480 |

## 5. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Provides solid baseline performance with good generalization. Performs reasonably well but is limited in capturing complex non-linear class boundaries. |
| Decision Tree | Achieves high accuracy and strong class separation. However, it may be prone to overfitting compared to ensemble methods. |
| K-Nearest Neighbors | Performance is moderate after feature scaling but lower than tree-based models. Sensitive to distance metric and choice of K. |
| Naive Bayes | Fast and computationally efficient but performs significantly lower than other models due to strong independence assumptions between features. |
| Random Forest (Ensemble) | Provides strong performance with improved generalization over a single Decision Tree. Effectively reduces overfitting and achieves high accuracy and MCC. |
| XGBoost (Ensemble) | Best performing model overall. Demonstrates highest accuracy, AUC, F1-score, and MCC. Effectively captures complex feature interactions and provides superior generalization. |

## 6. Streamlit Web Application

A Streamlit app was developed to:

- Upload test.csv
- Select model from dropdown
- Generate predictions
- Display evaluation metrics
- Display confusion matrix

To run the application locally:

```bash
pip install -r requirements.txt
streamlit run app.py