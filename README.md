# ML Assignment 2: Classification Models with Streamlit

## 1. Problem Statement

This assignment focuses on implementing and comparing multiple classification models on a chosen dataset, building an interactive Streamlit web application to demonstrate these models, and deploying the application. The goal is to gain hands-on experience with an end-to-end ML deployment workflow, including modeling, evaluation, UI design, and deployment.

## 2. Dataset Description

For this assignment, the **Breast Cancer Wisconsin (Diagnostic) dataset** was chosen from the UCI Machine Learning Repository (also available on Kaggle).

*   **Source**: UCI Machine Learning Repository
*   **Task**: Binary Classification
*   **Instances**: 569 samples
*   **Features**: 30 numerical features, which are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image.
*   **Target**: The target variable indicates whether the tumor is Malignant (1) or Benign (0).

This dataset meets the assignment requirements of having more than 500 instances and more than 12 features.

**Note**: The original dataset was imbalanced (357 Benign vs 212 Malignant). To ensure a uniform distribution for training, the minority class (Malignant) was upsampled to match the majority class count (357), resulting in a balanced dataset of 714 instances before splitting.

## 3. Models Used and Comparison

Six different classification models were implemented and evaluated on the preprocessed Breast Cancer Wisconsin dataset. The dataset was split into training and testing sets (80/20 split), and features were scaled using `StandardScaler`.

The following metrics were calculated for each model: Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

### Comparison Table of Evaluation Metrics

| ML Model Name          | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
| :--------------------- | :------- | :------- | :-------- | :------- | :------- | :------- |
| Logistic Regression    | 0.9859   | 0.9980   | 0.9859    | 0.9859   | 0.9859   | 0.9718   |
| Decision Tree          | 0.9648   | 0.9648   | 0.9583    | 0.9718   | 0.9650   | 0.9297   |
| KNN                    | 0.9296   | 0.9778   | 0.9420    | 0.9155   | 0.9286   | 0.8595   |
| Gaussian Naive Bayes   | 0.9648   | 0.9984   | 0.9459    | 0.9859   | 0.9655   | 0.9304   |
| Random Forest(Ensemble)| 0.9859   | 0.9994   | 0.9859    | 0.9859   | 0.9859   | 0.9718   |
| XGBoost (Ensemble)     | 0.9789   | 0.9978   | 0.9722    | 0.9859   | 0.9790   | 0.9578   |

### Observations on Model Performance

| ML Model Name          | Observation about model performance                                                                 |
| :--------------------- | :-------------------------------------------------------------------------------------------------- |
| Logistic Regression    | Tied for highest Accuracy (0.9859) and MCC (0.9718). Highly effective for this diagnostic task.    |
| Decision Tree          | Performed well (Accuracy 0.9648) but slightly lower than ensemble methods.                          |
| KNN                    | Lowest Accuracy (0.9296) and MCC (0.8595), suggesting distance-based clustering is less effective. |
| Naive Bayes            | High Recall (0.9859) and AUC (0.9984), outperforming KNN significantly.                             |
| Random Forest(Ensemble)| Top performer with near-perfect AUC (0.9994) and highest Accuracy/MCC.                             |
| XGBoost (Ensemble)     | Very strong performance (Accuracy 0.9789), competitive with Random Forest.                         |

*   **General Note**: The **Uniform Distribution** (resulting from upsampling) ensured balanced training, leading to stable metrics across the board. Ensemble methods and Logistic Regression proved most robust for this dataset.

## 4. Streamlit Application

The `app.py` file contains the Streamlit application. It allows users to:
*   Select one of the trained classification models from a dropdown.
*   View the evaluation metrics (Accuracy, AUC, Precision, Recall, F1 Score, MCC) for the selected model.
*   Visualize the confusion matrix for the selected model.

**To run the Streamlit app locally:**
1.  Ensure you have Python and the virtual environment set up (`.venv`).
2.  Activate the virtual environment: `source .venv/bin/activate`
3.  Run the app: `streamlit run app.py`

## 5. Repository Structure

```
project-folder/
|-- app.py                          # Streamlit web application
|-- requirements.txt                # Python dependencies
|-- README.md                       # This README file
|-- sample_test_data.csv            # Sample data for testing the application
|-- model/                          # Directory for trained models and data
|   |-- scaler.pkl                  # Scaler object
|   |-- X_test_scaled.pkl           # Scaled test features
|   |-- y_test.pkl                  # Test labels
|   |-- LogisticRegression.pkl      # Trained Logistic Regression model
|   |-- DecisionTreeClassifier.pkl  # Trained Decision Tree Classifier model
|   |-- KNeighborsClassifier.pkl    # Trained K-Nearest Neighbors Classifier model
|   |-- GaussianNB.pkl              # Trained Gaussian Naive Bayes model
|   |-- RandomForestClassifier.pkl  # Trained Random Forest Classifier model
|   |-- XGBClassifier.pkl           # Trained XGBoost Classifier model
|-- model_training.py               # Script to train and save models
|-- model_evaluation.py             # Script to evaluate models (generates CSV results)
|-- model_evaluation_results.csv    # CSV containing evaluation metrics for all models
```
