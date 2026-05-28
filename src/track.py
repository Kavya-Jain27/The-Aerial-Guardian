from ultralytics import YOLO
import supervision as sv
import cv2
import time

# Load YOLO model
model = YOLO("yolov8s.pt")

# Initialize ByteTrack
tracker = sv.ByteTrack()

# Video path
video_path = "data/sample.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output writer
out = cv2.VideoWriter(
    "outputs/tracking_output.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (width, height)
)

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

while True:
    start_time = time.time()
    ret, frame = cap.read()

    if not ret:
        break

    # Resize for better performance
    frame = cv2.resize(frame, (1280, 720))

    # YOLO detection
    results = model(
        frame,
        classes=[0],
        imgsz=1280,
        conf=0.20,
        verbose=False
    )[0]

    # Convert detections
    detections = sv.Detections.from_ultralytics(results)

    # Update tracker
    detections = tracker.update_with_detections(detections)

    # Labels
    labels = [
        f"ID {tracker_id}"
        for tracker_id in detections.tracker_id
    ]

    # Draw traces
    annotated_frame = trace_annotator.annotate(
        scene=frame.copy(),
        detections=detections
    )

    # Draw boxes
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    # Draw labels
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    # Resize back
    annotated_frame = cv2.resize(
        annotated_frame,
        (width, height)
    )

    out.write(annotated_frame)
    
    end_time = time.time()

    fps_value = 1 / (end_time - start_time)

    print(f"FPS: {fps_value:.2f}")

    print("Tracking persons...")

cap.release()
out.release()

print("Tracking completed successfully!")