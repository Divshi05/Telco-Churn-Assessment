# Telco Customer Churn Prediction

This project predicts customer churn for a telecom company using machine learning models. The dataset contains customer demographics, account information, and service usage data.

---

## **Dataset**

* Source: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
* Total entries: 7,043
* Columns: 21
  Key columns include:

  * `customerID`: Unique customer ID
  * `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  * `tenure`, `PhoneService`, `MultipleLines`
  * `InternetService`, `OnlineSecurity`, `OnlineBackup`
  * `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
  * `Contract`, `PaperlessBilling`, `PaymentMethod`
  * `MonthlyCharges`, `TotalCharges`
  * `Churn` (target variable: Yes/No)

---

## **Data Preprocessing**

1. **Handling Categories**

   * Replaced `"No internet service"` and `"No phone service"` with `"No"` in relevant columns.
   * Converted `TotalCharges` to numeric (filled errors with 0).
   * Dropped `customerID` column (irrelevant for modeling).
2. **Encoding**

   * One-Hot Encoding for categorical features like `gender`, `Partner`, `PaymentMethod`, etc.
   * Ordinal Encoding for `Contract` (`Month-to-month` < `One year` < `Two year`).
   * Target variable `Churn`: `Yes → 1`, `No → 0`.
3. **Feature Scaling**

   * Applied `StandardScaler` to numeric features.

---

## **Exploratory Data Analysis (EDA)**

* **Univariate Analysis:**

  * Count plots for categorical features, KDE for `MonthlyCharges`.
* **Bivariate Analysis:**

  * Churn rate depends on contract type, tenure, and payment method.
* **Multivariate Analysis:**

  * Correlation analysis showed strong correlation between `TotalCharges` and `tenure`.

---

## **Modeling**

Tested four classifiers:

| Model               | Train Accuracy | Test Accuracy | Notes               |
| ------------------- | -------------- | ------------- | ------------------- |
| Logistic Regression | 80.28%         | 81.97%        | Minimal overfitting |
| Decision Tree       | 99.86%         | 73.67%        | Overfitting         |
| Random Forest       | 99.86%         | 79.42%        | Overfitting         |
| Gradient Boosting   | 82.36%         | 81.26%        | Slight overfitting  |

* Observations: Logistic Regression and Gradient Boosting showed minimal overfitting. Others overfit the training data.

---

## **Hyperparameter Tuning**

* **Logistic Regression:** `solver='saga'`, `penalty='l1'`, `C=10`
* **Gradient Boosting:** `n_estimators=100`, `min_samples_split=20`, `max_depth=5`, `loss='exponential'`, `criterion='squared_error'`

**Post-Tuning Performance:**

| Model               | Train Accuracy | Test Accuracy | Notes            |
| ------------------- | -------------- | ------------- | ---------------- |
| Logistic Regression | 80.17%         | 81.97%        | Best balance     |
| Gradient Boosting   | 84.97%         | 80.55%        | Slightly overfit |

* **Conclusion:** Logistic Regression is the best-performing model after tuning.

---

## **Conclusion**

* Logistic Regression is the recommended model for predicting customer churn in this dataset.
* Key factors affecting churn include:

  * Contract type
  * Tenure
  * Payment method
* Gradient Boosting showed slightly better training performance but lower test performance, indicating slight overfitting.

---

## **Libraries Used**

* `numpy`, `pandas`, `matplotlib`, `seaborn`
* `scikit-learn` (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `StandardScaler`, `RandomizedSearchCV`)

---

## **How to Run**

1. Load the dataset: `df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')`
2. Run preprocessing steps to clean and encode data.
3. Split data into train/test sets and scale features.
4. Train models and evaluate using accuracy, F1-score, precision, and recall.
5. Apply hyperparameter tuning for the best models.

