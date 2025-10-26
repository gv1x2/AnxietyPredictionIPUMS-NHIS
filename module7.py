from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# Load your dataset
df = pd.read_csv('Book.csv', delimiter=';')

X = df.drop('ANXIETYEV', axis=1)  # Features
y = df['ANXIETYEV']  # Target variable

# Split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Random UnderSampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train_scaled, y_train)

print(f"Before undersampling: {X_train_scaled.shape[0]} instances, {sum(y_train == 0)} majority class, {sum(y_train == 1)} minority class")
print(f"After undersampling: {X_resampled.shape[0]} instances, {sum(y_resampled == 0)} majority class, {sum(y_resampled == 1)} minority class")

# Initialize and train the logistic regression model on the resampled training set
log_reg = LogisticRegression()
log_reg.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = log_reg.predict(X_test_scaled)

# Evaluate the model
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
