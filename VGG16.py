
# -*- coding: utf-8 -*-
#Bangash PhD Scholar
#Nov 2024

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

# Enable mixed precision to reduce memory usage
set_global_policy('mixed_float16')

# Paths
base_dir_1 = r' '
base_dir_2 = r' '
metadata_path = r'/HAM10000_metadata.csv'

# Load and preprocess metadata
metadata = pd.read_csv(metadata_path)
metadata_labels = metadata['dx']

# Encode labels and convert to one-hot encoding
label_encoder = LabelEncoder()
metadata_labels = label_encoder.fit_transform(metadata['dx'])
metadata_labels = to_categorical(metadata_labels, num_classes=7)  # Adjusted to 7 classes

# Preprocess metadata features
metadata_features = metadata[['age', 'sex', 'localization']]
metadata_features = pd.get_dummies(metadata_features)
scaler = StandardScaler()
metadata_features = scaler.fit_transform(metadata_features)

# Data augmentation with enhanced augmentation techniques
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generators for each directory
train_image_generator_1 = datagen.flow_from_directory(
    base_dir_1,
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

train_image_generator_2 = datagen.flow_from_directory(
    base_dir_2,
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_image_generator_1 = datagen.flow_from_directory(
    base_dir_1,
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

validation_image_generator_2 = datagen.flow_from_directory(
    base_dir_2,
    target_size=(64, 64),
    batch_size=8,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Function to combine batches from two generators and include metadata
def combine_generators(image_gen1, image_gen2, metadata_features):
    while True:
        # Get batches of images from both image generators
        image_batch1, label_batch1 = next(image_gen1)
        image_batch2, label_batch2 = next(image_gen2)
        
        # Combine images and labels
        images = np.concatenate([image_batch1, image_batch2])
        labels = np.concatenate([label_batch1, label_batch2])
        
        # Dynamically adjust metadata batch to match image batch size
        metadata_batch_size = images.shape[0]
        
        # Ensure metadata batch size matches images batch size
        if metadata_batch_size > metadata_features.shape[0]:
            metadata_batch = np.tile(metadata_features, (metadata_batch_size // metadata_features.shape[0] + 1, 1))
        else:
            metadata_batch = metadata_features

        metadata_batch = metadata_batch[:metadata_batch_size]  # Trim to match batch size

        # Yield as a tuple with image and metadata inputs
        yield [images, metadata_batch], labels

# Use the combine_generators function to create training and validation generators
train_gen = combine_generators(train_image_generator_1, train_image_generator_2, metadata_features)
val_gen = combine_generators(validation_image_generator_1, validation_image_generator_2, metadata_features)

# Building the VGG16 model with pretrained weights
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Freeze the VGG16 layers to retain pretrained features
for layer in vgg_base.layers:
    layer.trainable = False

# Add custom layers on top of VGG16 for classification
x = vgg_base.output
x = GlobalAveragePooling2D()(x)

# Building the metadata model with reduced units
metadata_input = Input(shape=(metadata_features.shape[1],))
y = Dense(32, activation='relu')(metadata_input)
y = Dense(16, activation='relu')(y)

# Concatenate image and metadata pathways
combined = Concatenate()([x, y])
z = Dense(64, activation='relu')(combined)
z = Dropout(0.6)(z)
output = Dense(7, activation='softmax', dtype='float32')(z)  # Updated to 7 units for 7 classes

# Final model with VGG16 as the base and custom layers for combined input
model = Model(inputs=[vgg_base.input, metadata_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-7, clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])

# Class weights to handle class imbalance (replace `class_weights_dict` with computed class weights)
class_weights_dict = {0: 1.0, 1: 2.0, 2: 1.5, 3: 1.2, 4: 1.0, 5: 1.8, 6: 1.3}  # Adjusted for 7 classes

# Callbacks for saving model, logging metrics, and stopping early with increased patience
model_save_path = r'C:/Users/Administrator/Downloads/ExperimentsforImages/Multi_Modal_Datasets/vgg16_custom_model_epoch.h5'
checkpoint_callback = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max')
csv_logger = CSVLogger('vgg16_custom_training_log.csv', append=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train the VGG16-based model for up to 20 epochs with early stopping and class weights
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    steps_per_epoch=(train_image_generator_1.samples + train_image_generator_2.samples) // 8,
    validation_steps=(validation_image_generator_1.samples + validation_image_generator_2.samples) // 8,
    callbacks=[checkpoint_callback, csv_logger, early_stopping],
    class_weight=class_weights_dict
)

# Save the final model
model.save(r'C:/Users/Administrator/Downloads/ExperimentsforImages/Multi_Modal_Datasets/vgg16_custom_model_64x64_final.h5')

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

# Print final metrics
print("Final training accuracy:", history.history['accuracy'][-1])
print("Final validation accuracy:", history.history['val_accuracy'][-1])
print("Training log saved to 'vgg16_custom_training_log.csv'")
