import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils.preprocessing import preprocess_tabular_data
import joblib

# Preprocess the tabular data
X_train, X_test, y_train, y_test, feature_names = preprocess_tabular_data('data/tabular/healthcare-dataset-stroke-data.csv')

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Tabular Model Accuracy: {accuracy * 100:.2f}%")

# Save the model and feature names
joblib.dump(model, 'models/tabular_model.pkl')
joblib.dump(feature_names, 'models/tabular_feature_names.pkl')