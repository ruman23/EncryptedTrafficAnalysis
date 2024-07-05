import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the dataset from a CSV file
df = pd.read_csv('data.csv')

# Drop the 'iteration' and 'maincategory' columns as they are not needed for classification
df = df.drop(['iteration', 'maincategory'], axis=1)

# Separate the features (X) and the target label (y)
X = df.drop('subcategory', axis=1)
y = df['subcategory']

# Encode the categorical target variable 'subcategory' into numeric values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training set
clf.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Prepare labels for the classification report
unique_labels = np.unique(np.concatenate((y_test, y_pred)))
class_names = label_encoder.inverse_transform(unique_labels)

# Generate and print the classification report
print(classification_report(y_test, y_pred, labels=unique_labels, target_names=class_names, zero_division=0))