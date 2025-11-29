import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATASET
filename = 'data_csv.csv'
print(f"Loading dataset from {filename}...")
# Treat '?' as NaN (common in datasets)
df = pd.read_csv(filename, na_values=['?'])

# 1.1 HANDLE MISSING VALUES
# SVM cannot handle missing numbers (NaN). We drop rows with any missing data.
original_count = len(df)
df.dropna(inplace=True)
print(f"📉 Dropped {original_count - len(df)} rows containing missing values.")
print(f"✅ Training with {len(df)} complete rows.")

# 2. DATA CLEANING (Standardizing Text)
# Sometimes CSVs have "Yes" and "yes" or " Asian" (with space). This fixes that.
categorical_cols = [
    'Depression', 'Anxiety_disorder', 'Sex', 'Ethnicity', 
    'Jaundice', 'Family_mem_with_ASD', 'Who_completed_the_test', 'ASD_traits'
]

for col in categorical_cols:
    # Convert to string, strip whitespace, and lower case for consistency, then title case
    df[col] = df[col].astype(str).str.strip().str.title() 

# Special fix: Ensure 'Middle Eastern' is consistent if it appears as 'middle eastern'
# The .title() above turns "middle eastern" into "Middle Eastern" automatically.

# 3. ENCODING (Converting Text to Numbers)
# We save the mappings so you know what numbers to use in your App/HTML
label_encoders = {}

print("\n" + "="*40)
print("🔠 FEATURE MAPPINGS (SAVE THESE FOR YOUR APP!)")
print("="*40)

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
    # Print the mapping for the user
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"\n🔹 {col} Mappings:")
    print(mapping)

print("\n" + "="*40)

# 4. DEFINE FEATURES AND TARGET
# Features (X) are everything except the target 'ASD_traits'
X = df.drop('ASD_traits', axis=1)
# Target (y) is 'ASD_traits'
y = df['ASD_traits']

# 5. SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. TRAIN SVM MODEL
print("\n🚀 Training SVM Model...")
svm_model = SVC(kernel='linear', probability=True) # Linear kernel is often good for this type of data
svm_model.fit(X_train, y_train)

# 7. EVALUATE
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. SAVE MODEL
# We verify the feature names are saved in the model
svm_model.feature_names_in_ = list(X.columns)

output_file = 'SVM_Final.pkl'
joblib.dump(svm_model, output_file)
print(f"\n💾 Model saved successfully as '{output_file}'")
print(f"Features expected by model: {list(X.columns)}")