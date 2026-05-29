from ultralytics import YOLO
import supervision as sv
import cv2
import os
import glob
import time

# Load model
model = YOLO("yolov8s.pt")

# ByteTrack
tracker = sv.ByteTrack()

# Path to image sequence
image_folder = "data/uav0000117_02622_v"

# Get all frames
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.*")))

# Read first frame
first_frame = cv2.imread(image_paths[0])

height, width, _ = first_frame.shape

# Output video writer
out = cv2.VideoWriter(
    "outputs/visdrone_tracking.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    20,
    (width, height)
)

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

prev_time = 0

for image_path in image_paths:

    frame = cv2.imread(image_path)

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # Resize for better detection
    frame_resized = cv2.resize(frame, (1280, 720))

    # YOLO detection
    results = model(
        frame_resized,
        classes=[0],
        imgsz=1280,
        conf=0.2,
        verbose=False
    )[0]

    # Convert detections
    detections = sv.Detections.from_ultralytics(results)

    # Tracking
    detections = tracker.update_with_detections(detections)

    # Labels
    labels = [
        f"ID {tracker_id}"
        for tracker_id in detections.tracker_id
    ]

    # Draw traces
    annotated_frame = trace_annotator.annotate(
        scene=frame_resized.copy(),
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

    # People count
    people_count = len(detections)

    cv2.putText(
        annotated_frame,
        f"People Count: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Resize back
    annotated_frame = cv2.resize(
        annotated_frame,
        (width, height)
    )

    out.write(annotated_frame)

    print(f"Processing {os.path.basename(image_path)}")

out.release()

print("VisDrone tracking completed!")