# Customer-Churn-prediction

This project aims to predict customer churn using machine learning techniques, leveraging the **Telco Customer Churn Dataset**. By identifying customers likely to churn, businesses can implement targeted retention strategies and reduce customer turnover.

---

## **Project Overview**
Customer churn refers to the rate at which customers stop doing business with a company. Predicting churn allows businesses to take proactive measures to retain customers, which is often more cost-effective than acquiring new ones.

This project evaluates multiple machine learning models, including Logistic Regression, Random Forest, and XGBoost, to find the most accurate model for predicting churn. Insights derived from the data help recommend actionable business strategies.

---

## **Dataset**
The dataset used for this project is the **Telco Customer Churn Dataset**. Key features include:

1. **Customer Demographics:** Gender, SeniorCitizen, Partner, Dependents.
2. **Account Information:** Tenure, Contract type, Payment method, MonthlyCharges, TotalCharges.
3. **Service Usage:** Internet service type, number of add-on services (e.g., StreamingTV, StreamingMovies).
4. **Target Variable:** Churn (Yes/No).

---

## **Project Structure**
```
Customer_Churn_Prediction/
│
├── README.md                 # Project description
├── churn_model.ipynb         # Jupyter Notebook with the project code
├── data/
│   └── telco_customer_churn.csv  # Dataset used (if shareable)
├── models/
│   └── model.pkl             # Saved model file (optional)
├── metrics/
│   └── Customer_Churn_Model_Metrics.csv  # Model metrics file
├── requirements.txt          # Python dependencies
├── LICENSE                   # License file (optional)
└── .gitignore                # To ignore unnecessary files (e.g., .DS_Store, __pycache__)
```

---

## **Installation Instructions**

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Customer_Churn_Prediction.git
   cd Customer_Churn_Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook to train and evaluate models:
   ```bash
   jupyter notebook churn_model.ipynb
   ```

---

## **Modeling Approach**
### **Steps**
1. **Data Preprocessing**:
   - Handled missing values.
   - Encoded categorical features.
   - Scaled numerical features using StandardScaler.

2. **Exploratory Data Analysis**:
   - Identified key churn drivers (e.g., contract type, monthly charges).
   - Visualized correlations and distributions.

3. **Model Evaluation**:
   - Tested Logistic Regression, Random Forest, and XGBoost.
   - Used GridSearchCV for hyperparameter tuning.

4. **Metrics**:
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC.

### **Results**
| Model                | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 80.2%    | 72.5%     | 65.3%  | 68.7%    | 77.5%   |
| Random Forest        | 84.7%    | 76.8%     | 71.2%  | 73.9%    | 83.1%   |
| XGBoost              | **86.5%**| **79.1%** | **75.3%**| **77.2%**| **85.9%**|

The **XGBoost Classifier** was selected as the final model due to its superior performance.

---

## **Key Findings and Recommendations**
1. **Key Predictors of Churn**:
   - Month-to-month contracts have the highest churn rates.
   - Customers with higher monthly charges are more likely to churn.
   - Longer-tenure customers are less likely to churn.

2. **Actionable Business Insights**:
   - Offer discounts to customers on month-to-month contracts to encourage long-term commitments.
   - Implement personalized retention strategies for high-risk customers.
   - Regularly engage with new customers to improve early-stage retention.

---

## **Future Work**
1. Include additional features such as customer feedback or social media sentiment.
2. Experiment with deep learning models for improved performance.
3. Develop a real-time churn prediction system.

---

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request for review.

---

## **License**
This project is licensed under the [MIT License](LICENSE).

---

## **Acknowledgments**
- Dataset: [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- Tools: Python, Pandas, Scikit-learn, XGBoost, Matplotlib
- Special thanks to the open-source community for their contributions!

