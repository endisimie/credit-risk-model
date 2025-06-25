# Credit Scoring Business Understanding.

1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Capital Accord requires financial institutions to quantify credit risk accurately and maintain adequate capital reserves based on the riskiness of their lending portfolio. Specifically, it promotes Internal Ratings-Based (IRB) approaches, which demand transparency in how credit risk models are developed, validated, and used.

This regulatory requirement places a strong emphasis on interpretability, audibility, and documentation. Stakeholders—including regulators, auditors, and internal compliance teams—must understand the rationale behind model predictions, the variables used, and their relationship to creditworthiness. Consequently, interpretable models (e.g., logistic regression with Weight of Evidence) are favored in production, even if they sacrifice some predictive performance, because they:

Allow for clear documentation and communication

Align with fairness and transparency standards

Can be validated and stress-tested more easily than black-box models

2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
In our current dataset, there is no explicit label indicating whether a customer defaulted on credit. However, to train a supervised machine learning model, a target variable is essential. We address this by creating a proxy variable—a rule-based surrogate for actual default—such as:

No repeat purchases after 90 days

Missed payments

High delay frequency in RFM patterns

While this approach enables us to begin model development, it introduces several risks:

Label noise: The proxy may incorrectly classify good customers as bad (or vice versa), reducing model accuracy.

Bias introduction: If the proxy is based on biased heuristics, it can lead to unfair decisions (e.g., penalizing low-income but timely customers).

Misalignment with real business outcomes: The proxy may not reflect actual loss or risk from a financial perspective, leading to misinformed loan decisions.

Thus, we must validate and refine our proxy using domain expertise, sensitivity testing, and eventually, real repayment data when available.

3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
Criteria	Simple (Logistic Regression + WoE)	Complex (Gradient Boosting / XGBoost)
Interpretability	High – easily explainable to regulators	Low – black-box nature requires SHAP/LIME for interpretation
Performance	May underperform on complex patterns	Typically higher predictive accuracy
Deployment	Straightforward to validate and monitor	Requires more robust MLOps infrastructure
Regulatory Compliance	Strong alignment with regulatory expectations	Requires justification and explanation tools
Scalability & Stability	Stable over time, fewer overfitting risks	Sensitive to overfitting, needs careful tuning and monitoring

Trade-off Summary:
In regulated environments like banking, the priority is often to use models that are transparent, fair, and auditable. While gradient boosting methods offer superior predictive power, they are best used either in hybrid models (e.g., for pre-screening) or supplemented with strong interpretability layers (e.g., SHAP). Logistic Regression with WoE remains the standard for its balance of performance and explainability.