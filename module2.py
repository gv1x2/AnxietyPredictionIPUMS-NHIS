import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('Book.csv', delimiter=';')  # Make sure the file name matches your actual file

# Separate the features (X) from the target variable (y)
X = df.drop('ANXIETYEV', axis=1)
y = df['ANXIETYEV']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to oversample the minority class in the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Initialize and train the logistic regression model on the resampled training set
log_reg_smote = LogisticRegression(max_iter=1000)
log_reg_smote.fit(X_train_smote, y_train_smote)

# Predict probabilities on the test set
y_pred_proba = log_reg_smote.predict_proba(X_test_scaled)[:, 1]  # Probabilities of the positive class

# Choose a new threshold
threshold = 0.3  # Lower than 0.5 to potentially increase the recall for the minority class

# Apply the new threshold to predict labels
y_pred_threshold = np.where(y_pred_proba > threshold, 1, 0)

# Calculate AUC-ROC and AUC-PR
auc_roc = roc_auc_score(y_test, y_pred_proba)
auc_pr = average_precision_score(y_test, y_pred_proba)

# Evaluate the model with the new threshold
print("Evaluations with New Threshold:")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_threshold))
print("\nClassification Report:\n", classification_report(y_test, y_pred_threshold))

# Print AUC-ROC and AUC-PR scores
print(f"\nAUC-ROC: {auc_roc}")
print(f"AUC-PR: {auc_pr}")
