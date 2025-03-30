# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 22:55:38 2024

@author: Administrator
"""
#Multimodal Skin Lesion Classification Using EfficientNetB0 with Image and Metadata Integration

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam

# Paths
base_dir_1 = r'C:/Users/Administrator/Downloads/ExperimentsforImages/Multi_Modal_Datasets/HAM10000_images_part_1'
base_dir_2 = r'C:/Users/Administrator/Downloads/ExperimentsforImages/Multi_Modal_Datasets/HAM10000_images_part_2'
metadata_path = r'C:/Users/Administrator/Downloads/ExperimentsforImages/Multi_Modal_Datasets/HAM10000_metadata.csv'

# Load and preprocess metadata
metadata = pd.read_csv(metadata_path)
label_encoder = LabelEncoder()
metadata_labels = label_encoder.fit_transform(metadata['dx'])
metadata_labels = to_categorical(metadata_labels, num_classes=7)

# Process metadata features
metadata_features = metadata[['age', 'sex', 'localization']]
metadata_features = pd.get_dummies(metadata_features)
scaler = StandardScaler()
metadata_features = scaler.fit_transform(metadata_features)

# Data augmentation and generators
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
train_image_generator_1 = datagen.flow_from_directory(
    base_dir_1, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training', shuffle=True
)
train_image_generator_2 = datagen.flow_from_directory(
    base_dir_2, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training', shuffle=True
)
validation_image_generator_1 = datagen.flow_from_directory(
    base_dir_1, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation', shuffle=False
)
validation_image_generator_2 = datagen.flow_from_directory(
    base_dir_2, target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation', shuffle=False
)

# Combine batches from two generators and include metadata
def combine_generators(image_gen1, image_gen2, metadata_features, labels):
    while True:
        image_batch1, label_batch1 = next(image_gen1)
        image_batch2, label_batch2 = next(image_gen2)
        images = np.concatenate([image_batch1, image_batch2])
        labels = np.concatenate([label_batch1, label_batch2])
        metadata_batch_size = images.shape[0]
        metadata_batch = np.tile(metadata_features, (metadata_batch_size // metadata_features.shape[0] + 1, 1))[:metadata_batch_size]
        yield [images, metadata_batch], labels

# Training and validation generators
train_gen = combine_generators(train_image_generator_1, train_image_generator_2, metadata_features, metadata_labels)
val_gen = combine_generators(validation_image_generator_1, validation_image_generator_2, metadata_features, metadata_labels)

# Model architecture
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Metadata input branch
metadata_input = Input(shape=(metadata_features.shape[1],))
y = Dense(32, activation='relu')(metadata_input)
y = Dense(16, activation='relu')(y)

# Concatenate image and metadata branches
combined = Concatenate()([x, y])
z = Dense(64, activation='relu')(combined)
z = Dropout(0.5)(z)
output = Dense(7, activation='softmax')(z)

# Build and compile the model
model = Model(inputs=[base_model.input, metadata_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
steps_per_epoch = (train_image_generator_1.samples + train_image_generator_2.samples) // 32
validation_steps = (validation_image_generator_1.samples + validation_image_generator_2.samples) // 32

model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps
)

# Evaluate the model and print classification metrics
y_true = []
y_pred = []

for i in range(validation_steps):
    # Get a batch of data
    [images, metadata_batch], labels = next(val_gen)
    predictions = model.predict([images, metadata_batch])
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

# Generate classification report
class_names = label_encoder.classes_
print(classification_report(y_true, y_pred, target_names=class_names))
