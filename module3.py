
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('Book.csv', delimiter=';')  # Ensure correct file name

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

# Initialize the Gradient Boosting model
#gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
#gb_clf.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
#y_pred = gb_clf.predict(X_test_scaled)

# Predict probabilities for AUC computation
#y_pred_proba = gb_clf.predict_proba(X_test_scaled)[:, 1]



# Initialize the Bagging classifier with a DecisionTreeClassifier as the base estimator
bagging_clf = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42)

# Train the model on the resampled training set
bagging_clf.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
y_pred = bagging_clf.predict(X_test_scaled)

# Predict probabilities for AUC computation
y_pred_proba = bagging_clf.predict_proba(X_test_scaled)[:, 1]


# Evaluate the model
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"\nAUC-ROC: {roc_auc_score(y_test, y_pred_proba)}")
print(f"AUC-PR: {average_precision_score(y_test, y_pred_proba)}")
