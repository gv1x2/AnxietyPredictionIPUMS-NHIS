import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE

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

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

### Stacking
estimators = [
    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42))
]
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

'''
### Blending (Simple implementation example)
# Split training set for blending
X_train_blend, X_val_blend, y_train_blend, y_val_blend = train_test_split(X_train_smote, y_train_smote, test_size=0.2, random_state=42)

# Base model
base_clf = RandomForestClassifier(n_estimators=10, random_state=42)
base_clf.fit(X_train_blend, y_train_blend)
val_predictions = base_clf.predict_proba(X_val_blend)[:, 1]

# Meta-model
clf = LogisticRegression()
clf.fit(val_predictions.reshape(-1, 1), y_val_blend)
'''

'''
### AdaBoost
clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)
'''

# Fit the model
clf.fit(X_train_smote, y_train_smote)

# Predictions
y_pred = clf.predict(X_test_scaled)
y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(f"\nAUC-ROC: {roc_auc_score(y_test, y_pred_proba)}")
print(f"AUC-PR: {average_precision_score(y_test, y_pred_proba)}")

