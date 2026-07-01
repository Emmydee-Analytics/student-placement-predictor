import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. LOAD AND PREPARE THE DATA
data_file = 'data.csv'


if not os.path.exists(data_file):
    raise FileNotFoundError(f"Could not find {data_file}")

df = pd.read_csv(data_file)
print("--- Dataset Loaded Successfully ---")
print(f"Dataset Size: {df.shape[0]} rows, {df.shape[1]} columns\n")

# Encode target variable: Placed = 1, Not Placed = 0
if 'placement_status' in df.columns:
    df['placement_status'] = df['placement_status'].map({'Placed': 1, 'Not Placed': 0})
else:
    df['placement_status'] = df['placement_status'].astype(int)

# Separate features (X) and target (y)
X = df.drop(columns=['placement_status'])
y = df['placement_status']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. MODEL TRAINING
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. EVALUATION
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("--- Model Performance ---")
print(f"Accuracy Score: {accuracy:.2%}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Placed', 'Placed']))

# 5. VISUALIZATION GENERATION
os.makedirs('plots', exist_ok=True)

# Plot 1: Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 5))
sns.barplot(x=importances[indices], y=features[indices], palette='viridis')
plt.title('Feature Importances in Predicting Student Placement')
plt.xlabel('Relative Importance Value')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('plots/feature_importance.png')
plt.close()

# Plot 2: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Placed', 'Placed'], yticklabels=['Not Placed', 'Placed'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png')
plt.close()

print("--- Analysis Complete! ---")
print("Saved analysis plots successfully inside the '/plots' folder.")
