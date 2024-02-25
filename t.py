import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.preprocessing import image

import numpy as np

batch_size = 32
model_path = "./model_enhanced.h5"

# Enhanced ImageDataGenerator with added data augmentation

datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=30,  # Random rotations

    width_shift_range=0.2,  # Random horizontal shifts

    height_shift_range=0.2,  # Random vertical shifts

    shear_range=0.2,  # Shear transformations

    zoom_range=0.2,  # Random zoom

    horizontal_flip=True,  # Horizontal flips

    validation_split=0.2  # Using 20% of the data for validation

)

 

# Load images from directories

train_generator = datagen.flow_from_directory(

    directory = '.',  # Assuming the current directory contains the 'smile' and 'non_smile' folders

    target_size = (150, 150),

    batch_size = 32,

    class_mode='binary',

    subset = 'training'

)

 

validation_generator = datagen.flow_from_directory(

    directory='.',  # Same as above

    target_size=(150, 150),

    batch_size = 32,

    class_mode='binary',

    subset='validation'

)

 

# Enhanced model with Dropout layers

model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),

    MaxPooling2D(2, 2),

    Dropout(0.25),  # Dropout layer

    Conv2D(64, (3,3), activation='relu'),

    MaxPooling2D(2, 2),

    Dropout(0.25),  # Dropout layer

    Conv2D(128, (3,3), activation='relu'),

    MaxPooling2D(2, 2),

    Flatten(),

    Dense(512, activation='relu'),

   Dropout(0.5),  # Increased dropout rate before the final layer

    Dense(1, activation='sigmoid')

])

 

# Compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

 

# Train the model with callbacks for early stopping and model checkpointing

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

 

history = model.fit(

    train_generator,

    steps_per_epoch=train_generator.samples // batch_size,

    epochs=30,  # Potentially increased number of epochs

    validation_data=validation_generator,

    validation_steps=validation_generator.samples // batch_size,

    verbose=2,

    callbacks=[early_stopping, model_checkpoint]

)

 

# Evaluate the model

val_loss, val_acc = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)

print(f'Validation accuracy: {val_acc}, Validation loss: {val_loss}')

 

# Prediction function remains the same

def predict_smile(image_path, model):

    img = image.load_img(image_path, target_size=(150, 150))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)  # Create a batch

    img_array /= 255.0  # Rescale pixel values to [0, 1]

 

    prediction = model.predict(img_array)

 

    return "Smiling" if prediction[0][0] >= 0.5 else "Not Smiling"

 

# Example usage

image_path = './smile/James_Jones_0001.jpg'

prediction = predict_smile(image_path, model)

print(prediction)
