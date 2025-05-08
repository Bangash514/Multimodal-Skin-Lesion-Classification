# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:52:50 2024

@author: Administrator
"""

#DenseNet

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Configure TensorFlow for optimal CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_soft_device_placement(True)

# Paths
base_dir_1 = r'path'
base_dir_2 = r'path'
metadata_path = r'path'

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
    rotation_range=10,  # Reduce rotation range to minimize computational cost
    width_shift_range=0.05,  # Reduce shift range to reduce memory usage
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,  # Reduce zoom range to decrease complexity
    brightness_range=[0.95, 1.05],
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generators with 128x128 image size
train_image_generator_1 = datagen.flow_from_directory(
    base_dir_1,
    target_size=(128, 128),
    batch_size=4,  # Reduce batch size to decrease memory usage
    class_mode='categorical',
    subset='training',
    shuffle=True
)

train_image_generator_2 = datagen.flow_from_directory(
    base_dir_2,
    target_size=(128, 128),
    batch_size=4,  # Reduce batch size to decrease memory usage
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_image_generator_1 = datagen.flow_from_directory(
    base_dir_1,
    target_size=(128, 128),
    batch_size=4,  # Reduce batch size to decrease memory usage
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

validation_image_generator_2 = datagen.flow_from_directory(
    base_dir_2,
    target_size=(128, 128),
    batch_size=4,  # Reduce batch size to decrease memory usage
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Combine batches from two generators and include metadata
def combine_generators(image_gen1, image_gen2, metadata_features):
    while True:
        image_batch1, label_batch1 = next(image_gen1)
        image_batch2, label_batch2 = next(image_gen2)
        images = np.concatenate([image_batch1, image_batch2])
        labels = np.concatenate([label_batch1, label_batch2])
        metadata_batch_size = images.shape[0]
        if metadata_batch_size > metadata_features.shape[0]:
            metadata_batch = np.tile(metadata_features, (metadata_batch_size // metadata_features.shape[0] + 1, 1))
        else:
            metadata_batch = metadata_features[:metadata_batch_size]
        yield [images, metadata_batch], labels

# Use the combine_generators function to create training and validation generators
train_gen = combine_generators(train_image_generator_1, train_image_generator_2, metadata_features)
val_gen = combine_generators(validation_image_generator_1, validation_image_generator_2, metadata_features)

# DenseNet121 with 128x128 input size
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Unfreeze more layers for fine-tuning
for layer in base_model.layers[-5:]:  # Further reduce the number of trainable layers to save memory
    layer.trainable = True

# Add custom layers on top of DenseNet
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Metadata input and layers
metadata_input = Input(shape=(metadata_features.shape[1],))
y = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(metadata_input)  # Reduce number of neurons to save memory
y = Dropout(0.3)(y)
y = Dense(8, activation='relu', kernel_regularizer=l2(0.001))(y)  # Reduce number of neurons

# Concatenate image and metadata pathways
combined = Concatenate()([x, y])
z = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(combined)  # Reduce number of neurons
z = Dropout(0.4)(z)  # Adjust dropout for better regularization
output = Dense(7, activation='softmax', dtype='float32')(z)

# Final model
model = Model(inputs=[base_model.input, metadata_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-5, clipnorm=1.0), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
model_save_path = r'C:/Users/Administrator/Downloads/ExperimentsforImages/Multi_Modal_Datasets/densenet_finetuned_model.h5'
checkpoint_callback = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max')
csv_logger = CSVLogger('densenet_training_log.csv', append=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)  # Reduce patience to stop earlier if no improvement
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)

# Train the DenseNet-based model for 10 epochs
history = model.fit(
    train_gen,
    epochs=10,  # Reduce epochs to fit within memory limits
    validation_data=val_gen,
    steps_per_epoch=(train_image_generator_1.samples + train_image_generator_2.samples) // 16,  # Adjust steps per epoch for new batch size
    validation_steps=(validation_image_generator_1.samples + validation_image_generator_2.samples) // 16,  # Adjust validation steps
    callbacks=[checkpoint_callback, csv_logger, early_stopping, lr_scheduler]
)

# Plot accuracy and loss
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

# Print final metrics
print("Final training accuracy:", history.history['accuracy'][-1])
print("Final validation accuracy:", history.history['val_accuracy'][-1])
print("Training log saved to 'densenet_training_log.csv'")
