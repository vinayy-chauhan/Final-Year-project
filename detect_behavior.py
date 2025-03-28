import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("models/crowd_behavior_model.h5")

# Class labels
CLASS_NAMES = ["Fight", "Running", "Walking"]

# Function to extract frames from video for prediction
def preprocess_video(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()
    while len(frames) < num_frames:
        frames.append(np.zeros((64, 64, 3)))

    return np.array(frames).reshape(1, num_frames, 64, 64, 3)

# Real-time behavior detection
def detect_behavior(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame sequence
        frames = preprocess_video(video_path)

        # Predict behavior
        prediction = model.predict(frames)
        behavior = CLASS_NAMES[np.argmax(prediction)]

        # Display results
        cv2.putText(frame, f"Detected: {behavior}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow("Crowd Behavior Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run detection on a video
video_path = "your_video.mp4"
detect_behavior(video_path)
