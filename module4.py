import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb

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

# Initialize the XGBoost model
# Calculate the scale_pos_weight value for imbalanced dataset
scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

xgb_clf = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss')

# Train the model on the resampled training set
xgb_clf.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
y_pred = xgb_clf.predict(X_test_scaled)

# Predict probabilities for AUC computation
y_pred_proba = xgb_clf.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"\nAUC-ROC: {roc_auc_score(y_test, y_pred_proba)}")
print(f"AUC-PR: {average_precision_score(y_test, y_pred_proba)}")

