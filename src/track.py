from ultralytics import YOLO
import supervision as sv
import cv2
import os
import glob
import time

# Load model
model = YOLO("yolov8m.pt")

# ByteTrack
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=90,
    minimum_matching_threshold=0.7
)

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
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)

prev_time = 0
fps_list = []
frame_count = 0


# -----------------------------
# TILING FUNCTION (QUAD SPLIT)
# -----------------------------
def get_tiles(frame):
    h, w, _ = frame.shape

    tiles = [
        frame[0:h//2, 0:w//2],      # top-left
        frame[0:h//2, w//2:w],      # top-right
        frame[h//2:h, 0:w//2],      # bottom-left
        frame[h//2:h, w//2:w]       # bottom-right
    ]

    coords = [
        (0, 0),
        (w//2, 0),
        (0, h//2),
        (w//2, h//2)
    ]

    return tiles, coords


# -----------------------------
# MAIN LOOP
# -----------------------------
for image_path in image_paths:
    frame_count += 1
    frame = cv2.imread(image_path)

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    if fps > 0:
        fps_list.append(fps)

    fps_display = sum(fps_list[-10:]) / len(fps_list[-10:]) if len(fps_list) > 0 else 0

    # enhance contrast
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)

    # -----------------------------
    # TILE-BASED DETECTION
    # -----------------------------
    tiles, coords = get_tiles(frame)

    all_detections = []

    for tile, (x_offset, y_offset) in zip(tiles, coords):

        results = model(
            tile,
            classes=[0],
            imgsz=1280,
            conf=0.10,
            verbose=False
        )[0]

        det = sv.Detections.from_ultralytics(results)

        # shift coordinates back to full frame
        if len(det) > 0:
            det.xyxy[:, [0, 2]] += x_offset
            det.xyxy[:, [1, 3]] += y_offset

            all_detections.append(det)

    # merge detections safely
    if len(all_detections) > 0:
        detections = sv.Detections.merge(all_detections)
    else:
        detections = sv.Detections.empty()


    # -----------------------------
    # TRACKING
    # -----------------------------
    detections = tracker.update_with_detections(detections)

    # Labels
    labels = []
    if detections.tracker_id is not None:
        labels = [f"Person {tid}" for tid in detections.tracker_id]

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    annotated_frame = trace_annotator.annotate(frame.copy(), detections)
    annotated_frame = box_annotator.annotate(annotated_frame, detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

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
        f"FPS: {fps_display:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    annotated_frame = cv2.resize(annotated_frame, (width, height))

    cv2.putText(
        annotated_frame,
        f"Frame: {frame_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    out.write(annotated_frame)

    print(f"Processing {os.path.basename(image_path)}")

out.release()
print("VisDrone tracking completed!")