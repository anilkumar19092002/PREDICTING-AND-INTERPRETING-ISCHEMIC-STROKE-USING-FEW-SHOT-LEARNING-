import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from tkinter import Tk, Label, Entry, Button, filedialog, messagebox, Frame, font

# Load models and feature names
tabular_model = joblib.load('models/tabular_model.pkl')
feature_names = joblib.load('models/tabular_feature_names.pkl')
image_model = tf.keras.models.load_model('models/image_model.keras')
image_accuracy = joblib.load('models/image_model_accuracy.pkl')  # Load hardcoded accuracy

# Function to predict stroke
def predict_stroke(patient_data, image_path, threshold=0.5):
    """
    Combine predictions from tabular and image models.
    """
    # Ensure patient_data has the correct feature names
    patient_data = pd.DataFrame([patient_data], columns=feature_names)

    # Predict using tabular model
    tabular_pred = tabular_model.predict_proba(patient_data)[0][1]  # Probability for class 1

    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Predict using image model
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    image_pred = image_model.predict(np.expand_dims(img_array, axis=0))[0][0]

    # Use hardcoded accuracy to weight the image model's prediction
    weighted_image_pred = image_pred * image_accuracy

    # Average predictions
    final_pred = (tabular_pred + weighted_image_pred) / 2

    # Determine if stroke is predicted
    stroke_prediction = "Yes" if final_pred >= threshold else "No"

    return final_pred, stroke_prediction

# Function to handle button click
def on_submit():
    try:
        # Get patient data from input fields
        patient_data = {
            'age': float(age_entry.get()),
            'hypertension': int(hypertension_entry.get()),
            'heart_disease': int(heart_disease_entry.get()),
            'avg_glucose_level': float(avg_glucose_level_entry.get()),
            'bmi': float(bmi_entry.get()),
            'gender_Male': int(gender_male_entry.get()),
            'ever_married_Yes': int(ever_married_yes_entry.get()),
            'work_type_Private': int(work_type_private_entry.get()),
            'Residence_type_Urban': int(residence_type_urban_entry.get()),
            'smoking_status_formerly smoked': int(smoking_status_formerly_smoked_entry.get()),
            'smoking_status_never smoked': int(smoking_status_never_smoked_entry.get()),
            'smoking_status_smokes': int(smoking_status_smokes_entry.get())
        }

        # Get image file path
        image_path = file_path_entry.get()

        # Predict stroke probability and outcome
        probability, prediction = predict_stroke(patient_data, image_path)

        # Display results in a colorful and attractive way
        result_text = f"Stroke Probability: {probability * 100:.2f}%\nStroke Prediction: {prediction}"
        result_label.config(text=result_text, fg="green" if prediction == "Yes" else "red", font=("Helvetica", 20, "bold"))

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to open file dialog
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    file_path_entry.delete(0, "end")
    file_path_entry.insert(0, file_path)

# Create the main window
root = Tk()
root.title("Stroke Prediction System")
root.geometry("600x700")
root.configure(bg="#f0f0f0")

# Create a frame for input fields
input_frame = Frame(root, bg="#f0f0f0")
input_frame.pack(pady=20)

# Labels and input fields for patient data
fields = [
    ("Age", "age_entry"),
    ("Hypertension (0 or 1)", "hypertension_entry"),
    ("Heart Disease (0 or 1)", "heart_disease_entry"),
    ("Average Glucose Level", "avg_glucose_level_entry"),
    ("BMI", "bmi_entry"),
    ("Gender Male (0 or 1)", "gender_male_entry"),
    ("Ever Married Yes (0 or 1)", "ever_married_yes_entry"),
    ("Work Type Private (0 or 1)", "work_type_private_entry"),
    ("Residence Type Urban (0 or 1)", "residence_type_urban_entry"),
    ("Smoking Status Formerly Smoked (0 or 1)", "smoking_status_formerly_smoked_entry"),
    ("Smoking Status Never Smoked (0 or 1)", "smoking_status_never_smoked_entry"),
    ("Smoking Status Smokes (0 or 1)", "smoking_status_smokes_entry")
]

for i, (label_text, entry_var) in enumerate(fields):
    Label(input_frame, text=label_text, bg="#f0f0f0", font=("Helvetica", 12)).grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = Entry(input_frame, font=("Helvetica", 12))
    entry.grid(row=i, column=1, padx=10, pady=5)
    globals()[entry_var] = entry  # Store entry widget in a global variable

# File path input
Label(input_frame, text="Image File Path", bg="#f0f0f0", font=("Helvetica", 12)).grid(row=len(fields), column=0, padx=10, pady=5, sticky="w")
file_path_entry = Entry(input_frame, font=("Helvetica", 12), width=30)
file_path_entry.grid(row=len(fields), column=1, padx=10, pady=5)
Button(input_frame, text="Browse", command=browse_file, font=("Helvetica", 12)).grid(row=len(fields), column=2, padx=10, pady=5)

# Submit button
submit_button = Button(root, text="Submit", command=on_submit, font=("Helvetica", 14, "bold"), bg="#4CAF50", fg="white")
submit_button.pack(pady=20)

# Result label
result_label = Label(root, text="", bg="#f0f0f0", font=("Helvetica", 20, "bold"))
result_label.pack(pady=20)

# Run the application
root.mainloop()