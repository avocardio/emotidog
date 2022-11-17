import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


data = tf.keras.utils.image_dataset_from_directory(
    'classes',
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=32,
    image_size=(512, 512),
    shuffle=True,
    seed=123
)

# Split the data into training and validation sets
train_size = int(0.8 * len(data))
val_size = int(0.2 * len(data))
train_data = data.take(train_size)
val_data = data.skip(train_size)

preprocessing = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomTranslation(0.1, 0.1)
])


# Create the model
model = tf.keras.Sequential([
    preprocessing,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 5, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 5, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 5, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50
)

# Save the model
model.save('model')

# Plot loss and accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()