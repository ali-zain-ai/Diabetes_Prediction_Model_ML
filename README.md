🚀 End-to-End Diabetes Prediction Model with Machine Learning
I worked on a healthcare-focused ML project where the objective was to predict whether a patient is diabetic or not based on clinical features such as glucose level, BMI, age, pregnancies, and more.

🛠️ Workflow & Approach

1️⃣ Data Collection & Understanding
Dataset: Pima Indians Diabetes Dataset
Explored distribution of diabetic vs non-diabetic patients
Performed statistical summary and feature analysis

2️⃣ Data Preprocessing
Separated features (X) and target labels (Y)
Applied StandardScaler to normalize features
Split data into training (80%) and testing (20%) sets

3️⃣ Model Training
Used Logistic Regression (max_iter=2500) to ensure convergence
Trained on standardized features for consistent performance

4️⃣ Model Evaluation
 📌 Training Data Results:
Accuracy: 77.52%
F1-score (Non-diabetic): 0.84
F1-score (Diabetic): 0.64
📌 Testing Data Results:
Accuracy: 77.92%
F1-score (Non-diabetic): 0.84
F1-score (Diabetic): 0.65

5️⃣ Insights
Model shows strong reliability in predicting non-diabetic cases (high recall = 0.90).
Diabetic case detection can be further improved by balancing the dataset, trying other algorithms (e.g., SVM, Random Forest, XGBoost), or tuning hyperparameters.

🔧 Tools & Libraries
Python (Scikit-learn, Pandas, NumPy)
Logistic Regression
StandardScaler

💡 Key Takeaway:
 This project demonstrates how ML can provide valuable insights in healthcare diagnostics, but it also highlights the importance of balancing model performance between positive and negative cases, especially in sensitive domains like diabetes prediction.
