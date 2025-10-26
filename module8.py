import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap

# Load your dataset
df = pd.read_csv('Book.csv', delimiter=';')

X = df.drop('ANXIETYEV', axis=1)  # Replace 'target_column' with your actual target column name
y = df['ANXIETYEV']

# Split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Coefficients in Linear Models
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
print("Coefficients in Linear Models:")
for i, v in enumerate(model.coef_[0]):
    print(f'Feature: {X.columns[i]}, Score: {v}')

'''
# Tree-based Feature Importance
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)
print("\nTree-based Feature Importance:")
for i, v in enumerate(model.feature_importances_):
    print(f'Feature: {X.columns[i]}, Score: {v}')

# Permutation Feature Importance
model = RandomForestClassifier().fit(X_train_scaled, y_train)  # Can use any fitted model
print("\nPermutation Feature Importance:")
results = permutation_importance(model, X_train_scaled, y_train, scoring='accuracy')
for i, v in enumerate(results.importances_mean):
    print(f'Feature: {X.columns[i]}, Score: {v}')

# SHAP Values
model = RandomForestClassifier().fit(X_train_scaled, y_train)  # SHAP works well with tree-based models
explainer = shap.Explainer(model, X_train_scaled)
shap_values = explainer.shap_values(X_train_scaled)
print("\nSHAP Values:")
shap.summary_plot(shap_values, X_train_scaled, feature_names=X.columns)
'''

