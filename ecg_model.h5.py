from tensorflow import keras
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Dense, Flatten, Dropout
from keras._tf_keras.keras.applications import VGG16
from keras._tf_keras.keras.preprocessing.image  import ImageDataGenerator

# Load the VGG16 model without the top layer (include_top=False)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model's layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Define the final model
ecg_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
ecg_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation data should not be augmented
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    'data/validation',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
# Train the model
ecg_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Save the fine-tuned model
ecg_model.save('fine_tuned_ecg_model.h5')
