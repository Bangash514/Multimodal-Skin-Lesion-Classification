import os
import shutil
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks

# Directories for training and validation sets
train_dir = r' '
val_dir = r' '

# Class names (replace with actual class names)
class_names = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6']

# Ensure directories for each class exist in both train and val folders
for class_name in class_names:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

# Define the ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load the images from the directories
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(299, 299),  # For InceptionResNetV2
                                                    batch_size=32,
                                                    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(299, 299),  # For InceptionResNetV2
                                                batch_size=32,
                                                class_mode='categorical')

# Check if the data is loaded correctly
print(f"Training set contains {train_generator.samples} samples.")
print(f"Validation set contains {val_generator.samples} samples.")

# Define the InceptionResNetV2 model with custom layers
base_model = InceptionResNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Dropout to avoid overfitting
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)  # Assuming len(class_names) is your number of classes
model = Model(inputs=base_model.input, outputs=predictions)
# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(train_generator, 
                    epochs=10, 
                    validation_data=val_generator, 
                    class_weight='auto',  # Handle class imbalance automatically
                    callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5)])

# Unfreeze some layers of the base model to fine-tune
for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers
    layer.trainable = True

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training the model
history = model.fit(train_generator, 
                    epochs=10, 
                    validation_data=val_generator, 
                    class_weight='auto')

# Save the model
model.save('model_inceptionresnetv2.h5')

# Evaluate the model
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Predict with the model
y_pred = model.predict(val_generator)
y_true = val_generator.classes

# calculate precision, recall, and F1-score
precision = precision_score(y_true, np.argmax(y_pred, axis=1), average='weighted')
recall = recall_score(y_true, np.argmax(y_pred, axis=1), average='weighted')
f1 = f1_score(y_true, np.argmax(y_pred, axis=1), average='weighted')

# Display Results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Accuracy: {val_accuracy:.4f}")
print(f"Final Validation Loss: {val_loss:.4f}")
