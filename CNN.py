import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load and preprocess data
csv_file_path = '/Users/anikshikarpuri/Desktop/CAPSTONE/petfinder-adoption-prediction/train/train.csv'
images_dir_path = '/Users/anikshikarpuri/Desktop/CAPSTONE/petfinder-adoption-prediction/train_images'
df = pd.read_csv(csv_file_path)

# Convert 'AdoptionSpeed' to categorical
num_classes = df['AdoptionSpeed'].nunique()
df['AdoptionSpeed'] = df['AdoptionSpeed'].apply(lambda x: to_categorical(x, num_classes=num_classes))

def load_image(image_path, target_size=(128, 128)):
    with Image.open(image_path) as img:
        img = img.convert('RGB').resize(target_size)
        return np.array(img) / 255.0

data, labels = [], []
for _, row in df.iterrows():
    pet_id, adoption_speed = row['PetID'], row['AdoptionSpeed']
    for image_name in os.listdir(images_dir_path):
        if image_name.startswith(pet_id):
            try:
                image = load_image(os.path.join(images_dir_path, image_name))
                data.append(image)
                labels.append(adoption_speed)
            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

data = np.array(data)
labels = np.array(labels)

# Splitting the dataset
X_temp, X_test, y_temp, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2222, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

# Training the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping, model_checkpoint])

# Load the best model and evaluate on the test set
model.load_weights('best_model.h5')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
