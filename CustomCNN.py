
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:47:09 2024

@author: Administrator
"""
#Bangash PhD scholar

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import gc
import psutil
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

# Enable mixed precision to reduce memory usage
set_global_policy('mixed_float16')

# Paths to image directories and metadata CSV file
base_dir = r'path'
metadata_path = os.path.join(base_dir, 'HAM10000_metadata.csv')

# Load and preprocess metadata
metadata = pd.read_csv(metadata_path)
metadata_labels = metadata['dx']

# Encode labels and convert to one-hot encoding
label_encoder = LabelEncoder()
metadata_labels = label_encoder.fit_transform(metadata_labels)
metadata_labels = np.clip(metadata_labels, 0, 5)
metadata_labels = to_categorical(metadata_labels, num_classes=6)

# Preprocess metadata features
metadata_features = metadata[['age', 'sex', 'localization']]
metadata_features = pd.get_dummies(metadata_features)
scaler = StandardScaler()
metadata_features = scaler.fit_transform(metadata_features)

# Data augmentation for images with reduced target size and batch size
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_image_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(64, 64),  # Reduced target size
    batch_size=8,  # Smaller batch size
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_image_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Building the custom CNN model
image_input = Input(shape=(64, 64, 3))

# First convolutional block
x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
x = MaxPooling2D((2, 2))(x)

# Second convolutional block
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

# Third convolutional block
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

# Fourth convolutional block
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = GlobalAveragePooling2D()(x)

# Building the metadata model
metadata_input = Input(shape=(metadata_features.shape[1],))
y = Dense(64, activation='relu')(metadata_input)
y = Dense(32, activation='relu')(y)

# Concatenate image and metadata pathways
combined = Concatenate()([x, y])
z = Dense(64, activation='relu')(combined)
z = Dropout(0.5)(z)
output = Dense(6, activation='softmax', dtype='float32')(z)

# Final model
model = Model(inputs=[image_input, metadata_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Custom generator to combine image and metadata inputs with matching batch sizes
def combined_generator(image_generator, metadata, labels, batch_size=8):
    while True:
        image_batch, image_labels = next(image_generator)
        batch_indices = image_generator.index_array[:image_batch.shape[0]]
        batch_indices = [i for i in batch_indices if i < len(metadata)]
        metadata_batch = metadata[batch_indices]
        label_batch = labels[batch_indices]

        # Ensure batch sizes are consistent
        if image_batch.shape[0] != metadata_batch.shape[0]:
            min_batch_size = min(image_batch.shape[0], metadata_batch.shape[0])
            image_batch = image_batch[:min_batch_size]
            metadata_batch = metadata_batch[:min_batch_size]
            label_batch = label_batch[:min_batch_size]

        yield [image_batch, metadata_batch], label_batch
        gc.collect()

train_metadata = np.array(metadata_features[:train_image_generator.samples])
train_labels = np.array(metadata_labels[:train_image_generator.samples])
validation_metadata = np.array(metadata_features[train_image_generator.samples:])
validation_labels = np.array(metadata_labels[train_image_generator.samples:])

# Create combined generators for training and validation
train_gen = combined_generator(train_image_generator, train_metadata, train_labels)
val_gen = combined_generator(validation_image_generator, validation_metadata, validation_labels)

# Callbacks for saving model, logging metrics, and stopping early
model_save_path = r'C:\Users\Administrator\Downloads\ExperimentsforImages\Deep_Learning\custom_cnn_model_epoch.h5'
checkpoint_callback = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max')
csv_logger = CSVLogger('custom_cnn_training_log.csv', append=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Train the custom CNN model for up to 20 epochs with early stopping
history = model.fit(
    train_gen,
    epochs=20,  # Set to 20 epochs
    validation_data=val_gen,
    steps_per_epoch=train_image_generator.samples // train_image_generator.batch_size,
    validation_steps=validation_image_generator.samples // validation_image_generator.batch_size,
    callbacks=[checkpoint_callback, csv_logger, early_stopping, tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch+1}: Memory usage: {psutil.virtual_memory().percent}%"))]
)

# Save the final model
model.save(r'C:\Users\Administrator\Downloads\ExperimentsforImages\Deep_Learning\custom_cnn_model_64x64_final.h5')

# Print final metrics after 20 epochs or early stopping
print("Final training accuracy:", history.history['accuracy'][-1])
print("Final validation accuracy:", history.history['val_accuracy'][-1])
print("Training log saved to 'custom_cnn_training_log.csv'")

# Plotting accuracy and loss over epochs
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()
