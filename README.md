#Credit Scoring Business Understanding
🔍 How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Capital Accord mandates that financial institutions quantify credit risk rigorously and maintain capital reserves proportionate to their risk exposure. Specifically, it encourages the Internal Ratings-Based (IRB) approach, which requires institutions to:

Build transparent and explainable models

Validate models regularly

Document modeling logic, assumptions, and performance

This regulatory framework places strong emphasis on interpretability and auditability. As a result, models like Logistic Regression with Weight of Evidence (WoE) are often preferred—even at the cost of some predictive performance—because they:

📝 Enable clear documentation and communication

⚖️ Align with fairness and transparency standards

🧪 Are easier to validate and stress test than black-box models

🧩 Why do we need a proxy target variable, and what are the risks?
Our dataset lacks a direct default label, which is essential for supervised learning. Therefore, we create a proxy variable to simulate high-risk behavior using heuristics such as:

❌ No repeat purchases within 90 days

📉 Low frequency or monetary activity (RFM patterns)

🚫 Missed or late payments

While this enables early model development, it introduces several business risks:

⚠️ Label noise: May misclassify good customers as bad (or vice versa), harming model precision

🧱 Bias risk: If the proxy is based on unfair rules, it may discriminate against certain customer groups

💼 Business misalignment: The proxy may not reflect actual financial losses, resulting in incorrect lending decisions

🛠️ Ongoing validation with domain experts and real repayment data is essential to refine this proxy.

| Criteria                   | Simple Model (Logistic Regression + WoE)    | Complex Model (Gradient Boosting / XGBoost)    |
| -------------------------- | ------------------------------------------- | ---------------------------------------------- |
| **Interpretability**       | ✅ High – easily explainable to stakeholders | ❌ Low – needs tools like SHAP/LIME             |
| **Predictive Power**       | ❌ Moderate – may miss non-linear patterns   | ✅ High – captures complex interactions         |
| **Deployment**             | ✅ Easy to validate and monitor              | ❌ Requires MLOps and explainability tools      |
| **Regulatory Fit**         | ✅ Strong compliance and auditability        | ⚠️ Needs extra documentation and justification |
| **Stability & Robustness** | ✅ Less prone to overfitting                 | ⚠️ Sensitive to data drift, needs tuning       |



This project builds an end-to-end credit scoring solution for **Bati Bank** in collaboration with an eCommerce partner offering a **Buy-Now-Pay-Later** service. The system leverages behavioral RFM (Recency, Frequency, Monetary) data to estimate credit risk and support responsible lending decisions.

---

## 📁 Project Structure

```
credit-risk-model/
├── data/                    # Raw and processed data
├── models/                  # Trained models and pipelines
├── notebooks/               # Jupyter notebooks for EDA
├── src/                     # Core Python source code
│   ├── api/                 # FastAPI app files
│   ├── feature_engineering_pipeline.py
│   ├── proxy_target_engineering.py
│   ├── train.py             # Model training & MLflow tracking
├── tests/                  # Unit tests for feature pipeline
├── .github/workflows/      # GitHub Actions CI config
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚦 Task Overview

### ✅ Task 1 – Credit Scoring Business Understanding

A deep dive into Basel II regulations, the importance of interpretability in risk models, and trade-offs between simple (e.g., Logistic Regression + WoE) and complex models (e.g., Gradient Boosting).

> See section: [📘 Credit Scoring Business Understanding](#credit-scoring-business-understanding)

---

### 📊 Task 2 – Exploratory Data Analysis (EDA)
- Summary statistics and data types
- Distribution of numerical and categorical variables
- Correlation matrix
- Missing values & outliers
- Key insights saved and visualized in Jupyter Notebook

---

### 🛠️ Task 3 – Feature Engineering
- Built a `sklearn.pipeline.Pipeline` for reproducibility
- Features created:
  - Aggregate: `TotalAmount`, `TransactionCount`, `AverageAmount`, `AmountStdDev`
  - Temporal: `TransactionHour`, `TransactionDay`, `TransactionMonth`, `TransactionYear`
- Handled missing values and standardization
- Encoded categorical variables using OneHotEncoder and LabelEncoder
- All logic implemented in `src/feature_engineering_pipeline.py`

---

### 🎯 Task 4 – Proxy Target Variable Engineering
- Used RFM metrics to define proxy target `is_high_risk`
- Applied KMeans clustering to segment customers into 3 groups
- Customers in the least active segment labeled as `is_high_risk = 1`
- Merged proxy back into dataset for training

---

### 🤖 Task 5 – Model Training & Tracking
- Trained two models: `Logistic Regression`, `Gradient Boosting Classifier`
- Metrics logged: Accuracy, Precision, Recall, F1, ROC-AUC
- Visualized:
  - Confusion Matrix
  - ROC Curve
- Used **MLflow** to track:
  - Parameters, metrics
  - Model versions and artifacts
- Best model saved to `models/best_model.joblib`

---

### 🔧 Task 6 – Model Deployment & CI/CD
- Created REST API with **FastAPI**:
  - `POST /predict`: Takes customer features and returns `risk_probability`
- Used **Pydantic** for request/response schema validation
- Containerized with Docker and `uvicorn`
- GitHub Actions CI pipeline:
  - Runs `flake8` for linting
  - Runs `pytest` to ensure test coverage
- Project passes all CI checks before deployment

---

## 🚀 How to Run

### 1. Set up the Environment
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run the API
```bash
uvicorn src.api.main:app --reload
```

### 3. Docker Deployment
```bash
docker-compose up --build
```

### 4. Run Tests and Linter
```bash
pytest tests/ -v
```

---

## 📘 Credit Scoring Business Understanding
(See earlier section for details on Basel II, interpretability, proxy risk design, and model trade-offs)

---
---

**Project Status:** ✅ MVP complete. Model trained, deployed, and tracked.

Feel free to fork, extend with real repayment labels, or connect to live APIs for production integration.
