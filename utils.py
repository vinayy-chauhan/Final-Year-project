import cv2
import numpy as np

# Extract frames from a video for training/testing
def extract_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))  # Resize to match model input
        frame = frame / 255.0  # Normalize pixel values
        frames.append(frame)

    cap.release()

    # Pad with blank frames if video is too short
    while len(frames) < num_frames:
        frames.append(np.zeros((64, 64, 3)))

    return np.array(frames)

# Preprocess video for model prediction
def preprocess_video(video_path, num_frames=30):
    frames = extract_frames(video_path, num_frames)
    return frames.reshape(1, num_frames, 64, 64, 3)

# Detect people in a frame (for density calculation)
def count_people(frame):
    people_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    people = people_cascade.detectMultiScale(gray, 1.1, 3)

    return len(people), people  # Return number of people and their bounding boxes

# Overlay detected people on video
def draw_people_boxes(frame, people):
    for (x, y, w, h) in people:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green boxes around people
    return frame
