import pandas as pd
import cv2
import numpy as np
from scipy.spatial import KDTree
import mediapipe as mp

# Load dataset
df = pd.read_csv("dataset.csv", header=None)
df.columns = ["x", "y", "width", "height", "label"]

# Convert features and labels
X = df.iloc[:, :-1].values  # Features: x, y, width, height
labels = np.array(df.iloc[:, -1].values)  # Labels (renamed from y to labels)

# Build a KDTree for fast nearest neighbor search
tree = KDTree(X)

# Label color mapping
label_colors = {
    "Relaxed": (0, 255, 0),   # Green
    "Steady": (0, 255, 255),  # Yellow
    "Intense": (0, 0, 255)    # Red
}

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB (Mediapipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape

                # Bounding box coordinates
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                # Find the closest match in dataset
                _, index = tree.query([[x, y, width, height]])  # Nearest neighbor search

                # Ensure index is an integer
                index = int(index[0])

                # Ensure index is within bounds
                if index >= labels.shape[0]:  
                    print(f"Warning: Index {index} is out of bounds!")
                    continue  

                pred_label = labels[index]  # Get the predicted label

                # Get the color for the label
                box_color = label_colors.get(pred_label, (255, 255, 255))  # Default to white if unknown

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + width, y + height), box_color, 2)
                cv2.putText(frame, pred_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        # Show output
        cv2.imshow("Intensity Detector", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
