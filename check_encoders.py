import joblib
import pandas as pd

# Load the label encoders
try:
    label_encoders = joblib.load('label_encoders.sav')
    print("Label Encoders loaded successfully!")
    print("Available encoders:", list(label_encoders.keys()))
    
    for feature, encoder in label_encoders.items():
        print(f"\n{feature.upper()} encoder classes:")
        print(f"  Classes: {list(encoder.classes_)}")
        print(f"  Example mappings:")
        for i, class_name in enumerate(encoder.classes_[:5]):  # Show first 5
            print(f"    '{class_name}' -> {i}")
        if len(encoder.classes_) > 5:
            print(f"    ... and {len(encoder.classes_) - 5} more")
            
except Exception as e:
    print(f"Error loading encoders: {e}")

# Also check target encoder
try:
    target_encoder = joblib.load('target_encoder.sav')
    print(f"\nTarget encoder classes: {list(target_encoder.classes_)}")
except Exception as e:
    print(f"Error loading target encoder: {e}")
