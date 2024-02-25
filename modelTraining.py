# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.preprocessing import image
# import numpy as np

# import cv2
# import mediapipe as mp
# import time
# from tensorflow.keras.models import load_model
# import numpy as np

# model_path = "./model.h5"
# # Define paths to your datasets

# smiling_images_path = './smile'
# non_smiling_images_path = './non_smile'


# # Create an ImageDataGenerator for data augmentation and normalization

# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2  # using 20% of the data for validation
# )



# # Specify the size of your images and batch size
# img_width, img_height = 150, 150
# batch_size = 32


# # Load images from directories
# train_generator = datagen.flow_from_directory(
#     directory='.',  # This should be the parent directory of your 'smiling' and 'non_smiling' folders
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='training'
# )

 
# validation_generator = datagen.flow_from_directory(
#     directory='.',  # This should be the parent directory of your 'smiling' and 'non_smiling' folders
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='validation'
# )
 

# # Define the model
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(img_width, img_height, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(128, (3,3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])


# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# # Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=10,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size,
#     verbose=2
# )


# # Evaluate the model (optional)
# val_loss, val_acc = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
# print(f'Validation accuracy: {val_acc}, Validation loss: {val_loss}')
# model.save(model_path)
 
# # Function to preprocess and predict the image

# def predict_smile(image_path, model):
#     # Load and preprocess the image
#     img = image.load_img(image_path, target_size=(150, 150))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Create a batch
#     img_array /= 255.0  # Rescale pixel values to [0, 1]

 

#     # Make a prediction
#     prediction = model.predict(img_array)


#     # Interpret the prediction
#     if prediction[0][0] >= 0.5:
#         return "Smiling"

#     else:
#         return "Not Smiling"

 

# # Example usage
# image_path = './smile/James_Jones_0001.jpg'
# prediction = predict_smile(image_path, model)
# print(prediction)


# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2  # using 20% of the data for validation
# )

#changed to enhanced if we want
model = load_model("./model.h5")

def predict_smile(image_path, model, datagen):
    # Load and preprocess the image using the same data generator used for training
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    
    # Use the same preprocessing as in the ImageDataGenerator for training
    img_array = datagen.standardize(img_array)

    # Make a prediction
    prediction = model.predict(img_array)

    # Interpret the prediction
    if prediction[0][0] >= 0.5:
        return "Smiling"
    else:
        return "Not Smiling"
    
image_path = 'non_smile/Aaron_Eckhart_0001.jpg'
prediction = predict_smile(image_path, model,datagen)
print(prediction)
