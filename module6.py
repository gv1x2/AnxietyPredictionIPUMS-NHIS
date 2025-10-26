import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import AUC

# Load the dataset
df = pd.read_csv('Book.csv', delimiter=';')  # Adjust path as necessary

# Separate the features (X) from the target variable (y)
X = df.drop('ANXIETYEV', axis=1)
y = df['ANXIETYEV']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Define the model
model = Sequential([
    Dense(64, input_dim=X_train_smote.shape[1], activation='relu'),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])

# Train the model
history = model.fit(X_train_smote, y_train_smote, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=32, verbose=2)

# Evaluate the model on the test data
evaluation = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}, Test AUC: {evaluation[2]}")

