from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8s.pt")

# Input video
video_path = "data/sample.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = 20

# Output video
out = cv2.VideoWriter(
    "outputs/detection_output.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (width, height)
)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Resize frame smaller for faster inference
    frame = cv2.resize(frame, (1280, 720))

    # Run detection
    results = model(
        frame,
        classes=[0],
        imgsz=1280,
        conf=0.20,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Resize back to original size
    annotated_frame = cv2.resize(
        annotated_frame,
        (width, height)
    )

    out.write(annotated_frame)

    print(f"Processed frame: {frame_count}")

cap.release()
out.release()

print("Video saved successfully!")