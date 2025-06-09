# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 10:25:39 2025

@author: Administrator
"""
#Bangash PhD Scholar

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight

# === Step 1: Updated Paths ===
image_dirs = [
    "C:/Users/Administrator/Downloads/ExperimentsforImages/Deep_Learning/2025/Datasets/HAM10000_images_part_1",
    "C:/Users/Administrator/Downloads/ExperimentsforImages/Deep_Learning/2025/Datasets/HAM10000_images_part_2"
]
metadata_path = "C:/Users/Administrator/Downloads/ExperimentsforImages/Deep_Learning/2025/Datasets/HAM10000_metadata.csv" 

# === Step 2: Load Data ===
metadata = pd.read_csv(metadata_path)
metadata['image_path'] = metadata['image_id'].apply(lambda x: [os.path.join(d, x + '.jpg') for d in image_dirs])
metadata['image_path'] = metadata['image_path'].apply(lambda x: next((img for img in x if os.path.exists(img)), None))
metadata = metadata.dropna(subset=['image_path'])

# === Step 3: Preprocess Image Data ===
img_height, img_width = 256, 256

# Increased Data Augmentation for better generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Increased rotation range
    width_shift_range=0.2,  # Slightly higher width shift
    height_shift_range=0.2,  # Slightly higher height shift
    shear_range=0.2,  # Increased shear range
    zoom_range=0.2,  # Increased zoom range
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.7, 1.3],  # More aggressive brightness range
    channel_shift_range=20.0,  # More channel shift
    validation_split=0.2
)

# Load images using flow_from_dataframe
train_gen = train_datagen.flow_from_dataframe(
    dataframe=metadata,
    directory=None,
    x_col='image_path',
    y_col='dx',
    target_size=(img_height, img_width),
    batch_size=16,  # Reduced batch size for smoother performance on laptop
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_dataframe(
    dataframe=metadata,
    directory=None,
    x_col='image_path',
    y_col='dx',
    target_size=(img_height, img_width),
    batch_size=16,  # Reduced batch size for smoother performance on laptop
    class_mode='categorical',
    subset='validation'
)

# === Step 4: Label Encoding ===
label_encoder = LabelEncoder()
metadata['dx'] = label_encoder.fit_transform(metadata['dx'])

# === Step 5: Define the Model ===
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

base_model.trainable = True
fine_tune_at = 30  # Fine-tune from the 30th layer onward
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.6),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile with adjusted learning rate
learning_rate = 0.00005  # Lower learning rate for stable convergence
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Step 6: Handle Class Imbalance ===
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(metadata['dx']), y=metadata['dx'])
class_weight_dict = dict(enumerate(class_weights))

# === Step 7: Set Callbacks ===
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Reduced patience
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)  # Save best model

# === Step 8: Train the Model ===
epochs = 30  # Reduced to 30 epochs for faster training
history = model.fit(
    train_gen,
    epochs=epochs,  # Train for 30 epochs
    validation_data=val_gen,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, model_checkpoint]
)

# === Step 9: Save the Model ===
model_save_path = "C:/Users/Administrator/Downloads/ExperimentsforImages/Deep_Learning/2025/saved_model.keras"
model.save(model_save_path)  # Save the final model
print(f"Model saved to: {model_save_path}")

# === Step 10: Print Training and Validation Metrics ===
train_accuracy = history.history['accuracy'][-1]
train_loss = history.history['loss'][-1]
val_accuracy = history.history['val_accuracy'][-1]
val_loss = history.history['val_loss'][-1]

print(f"Total Training Accuracy: {train_accuracy:.4f}")
print(f"Total Training Loss: {train_loss:.4f}")
print(f"Total Validation Accuracy: {val_accuracy:.4f}")
print(f"Total Validation Loss: {val_loss:.4f}")

# === Step 11: Evaluate the Model ===
val_preds = model.predict(val_gen)
y_pred = np.argmax(val_preds, axis=1)

# Print Classification Report
print("\nClassification Report:\n")
print(classification_report(val_gen.classes, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(val_gen.classes, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.show()

# === Optional: Plot Training History ===
plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
