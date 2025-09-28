# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 08:16:10 2025

@author: Administrator
"""

#DenseNet169_New
#Bangash ..


# DenseNet169 Model with Enhanced Metrics


import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import psutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import DenseNet169  # Use DenseNet169 instead of DenseNet121
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc

# Configure TensorFlow for optimal CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_soft_device_placement(True)

# Paths
image_directory = '.../HAM10000_images'  # Updated the directory path for images
metadata_path = '.../HAM10000_metadata.csv'  # Path for metadata

# Load and preprocess metadata
metadata = pd.read_csv(metadata_path)
metadata_labels = metadata['dx']

# Encode labels and convert to one-hot encoding
label_encoder = LabelEncoder()
metadata_labels = label_encoder.fit_transform(metadata['dx'])
metadata_labels = to_categorical(metadata_labels, num_classes=7)

# Preprocess metadata features
metadata_features = metadata[['age', 'sex', 'localization']]
metadata_features = pd.get_dummies(metadata_features)
scaler = StandardScaler()
metadata_features = scaler.fit_transform(metadata_features)

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=30,  # Increase rotation range to improve data variability
    width_shift_range=0.15,  # Increase shift range to improve model robustness
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,  # Increase zoom range for better generalization
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Image generators (with 128x128 image size)
train_image_generator = datagen.flow_from_directory(
    image_directory,
    target_size=(128, 128),  # 128x128 target size
    batch_size=64,  # Increase batch size for more stable training
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_image_generator = datagen.flow_from_directory(
    image_directory,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Combine image batches with metadata
def combine_generators(image_gen, metadata_features):
    while True:
        image_batch, label_batch = next(image_gen)
        batch_indices = np.arange(image_batch.shape[0])
        metadata_batch = metadata_features[batch_indices % metadata_features.shape[0]]
        yield [image_batch, metadata_batch], label_batch
        gc.collect()

# Combine image and metadata for training and validation
train_gen = combine_generators(train_image_generator, metadata_features)
val_gen = combine_generators(validation_image_generator, metadata_features)

# DenseNet169 with 128x128 input size
image_input = Input(shape=(128, 128, 3))
base_model = DenseNet169(weights='imagenet', include_top=False, input_tensor=image_input)  # Use DenseNet169

# Freeze initial layers in DenseNet169 to use as a feature extractor
for layer in base_model.layers[:-20]:  # Freeze fewer layers to allow more feature learning
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Add custom layers on top of DenseNet169
x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)  # Regularization to reduce overfitting
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

# Metadata input and layers
metadata_input = Input(shape=(metadata_features.shape[1],))
y = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(metadata_input)
y = BatchNormalization()(y)
y = Dropout(0.2)(y)
y = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(y)
y = BatchNormalization()(y)

# Concatenate image and metadata pathways
combined = Concatenate()([x, y])
z = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(combined)
z = BatchNormalization()(z)
z = Dropout(0.3)(z)
output = Dense(7, activation='softmax', dtype='float32')(z)

# Final model
model = Model(inputs=[base_model.input, metadata_input], outputs=output)
model.compile(optimizer=SGD(learning_rate=1e-5, momentum=0.9, clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
def lr_schedule(epoch, lr):
    if epoch > 10:
        return lr * 0.5
    return lr

# Callbacks
checkpoint_dir = 'C:/Users/Administrator/Downloads/ExperimentsforImages/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Save the model at regular intervals
checkpoint_callback = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}.h5'),
    save_freq='epoch',  # Save after every epoch
    save_best_only=False,  # Save at every epoch
    monitor='val_accuracy',  # Monitor validation accuracy
    mode='max',  # Keep the best model based on validation accuracy
    verbose=1
)

# Additional callbacks
csv_logger = CSVLogger('densenet169_training_log_128.csv', append=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)  # Allow more training
lr_scheduler = LearningRateScheduler(lr_schedule)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)  # Convert one-hot to class labels
    y_pred = np.argmax(y_pred, axis=1)  # Convert one-hot to class labels

    # Precision, recall, f1-score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report
    report = classification_report(y_true, y_pred)

    # ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')
    pr_auc = average_precision_score(y_true, y_pred, average='weighted', multi_class='ovr')

    # Support per class
    support = np.bincount(y_true)

    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{report}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Support: {support}")
    print(f"MCC: {mcc:.4f}")

    return precision, recall, f1, cm, report, roc_auc, pr_auc, support, mcc

# Train the model
history = model.fit(
    train_gen,
    epochs=15,
    validation_data=val_gen,
    steps_per_epoch=300,  # Adjust based on data
    validation_steps=75,  # Adjust based on data
    callbacks=[checkpoint_callback, csv_logger, early_stopping, lr_scheduler],
    verbose=1
)

# Print final metrics after training
y_true = np.concatenate([y for _, y in val_gen], axis=0)
y_pred = model.predict(val_gen, steps=len(validation_image_generator))

# Call the function to calculate metrics
calculate_metrics(y_true, y_pred)

# Final metrics
print("Final training accuracy:", history.history['accuracy'][-1])
print("Final validation accuracy:", history.history['val_accuracy'][-1])
print("Final training loss:", history.history['loss'][-1])
print("Final validation loss:", history.history['val_loss'][-1])

# Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()

print("Training log saved to 'densenet169_training_log_128.csv'")
