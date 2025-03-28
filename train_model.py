import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Dataset path (contains "fight", "run", "walk" folders)
DATASET_PATH = "dataset/"
NUM_FRAMES = 30  # Number of frames per video

# Function to extract frames from videos
def extract_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))
        frame = frame / 255.0  # Normalize
        frames.append(frame)

    cap.release()
    while len(frames) < num_frames:
        frames.append(np.zeros((64, 64, 3)))  # Pad with blank frames if needed
    
    return np.array(frames)

# Load dataset
X, y = [], []
labels = {"fight": 0, "run": 1, "walk": 2}
for category in labels.keys():
    folder = os.path.join(DATASET_PATH, category)
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            video_path = os.path.join(folder, file)
            frames = extract_frames(video_path, NUM_FRAMES)
            X.append(frames)
            y.append(labels[category])

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split into training & validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN-LSTM Model
model = Sequential([
    TimeDistributed(Conv2D(16, (3, 3), activation='relu'), input_shape=(NUM_FRAMES, 64, 64, 3)),
    TimeDistributed(MaxPooling2D(2, 2)),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dense(3, activation='softmax')  # 3 classes: Fight, Run, Walk
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train the model
print("Training model...")
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# Save the model
model.save("models/crowd_behavior_model.h5")
print("Model trained and saved successfully!")
