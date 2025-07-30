import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load your data (replace with your actual data loading code)
# df = pd.read_csv('your_data.csv')
# X = df.drop('target_column', axis=1)
# y = df['target_column']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to try
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Create pipelines for each model
pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('classifier', model)
    ])

# Evaluate models using cross-validation
print("Model Evaluation using Cross-Validation:")
print("-" * 50)
for name, pipeline in pipelines.items():
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"{name}:")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})\n")

# Train the best performing model (Random Forest) on full training data
best_model = pipelines['Random Forest']
best_model.fit(X_train, y_train)

# Evaluate on test set
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("\nBest Model (Random Forest) Performance:")
print("-" * 50)
print(f"Test Accuracy: {test_accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Get feature importances
if isinstance(best_model.named_steps['classifier'], RandomForestClassifier):
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    feature_names = best_model.named_steps['poly'].get_feature_names_out()
    importance_dict = dict(zip(feature_names, feature_importances))
    
    print("\nTop 10 Most Important Features:")
    print("-" * 50)
    for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{feature}: {importance:.3f}")

# Save the best model
import pickle
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f) 