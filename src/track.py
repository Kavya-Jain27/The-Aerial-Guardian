from ultralytics import YOLO
import supervision as sv
import cv2
import os
import glob
import time
import json

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("yolov8m.pt")

# -----------------------------
# BYTE TRACKER
# -----------------------------
tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=90,
    minimum_matching_threshold=0.7
)

# -----------------------------
# IMAGE SEQUENCE PATH
# -----------------------------
image_folder = "data/uav0000117_02622_v"

# Get all frames
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.*")))

# Read first frame
first_frame = cv2.imread(image_paths[0])

height, width, _ = first_frame.shape

# -----------------------------
# OUTPUT VIDEO
# -----------------------------
out = cv2.VideoWriter(
    "outputs/visdrone_tracking.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    20,
    (width, height)
)

# -----------------------------
# ANNOTATORS
# -----------------------------
box_annotator = sv.BoxAnnotator(thickness=2)

label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1
)

trace_annotator = sv.TraceAnnotator(
    thickness=2,
    trace_length=30
)

# -----------------------------
# FPS VARIABLES
# -----------------------------
prev_time = 0
fps_list = []

# -----------------------------
# FRAME COUNTER
# -----------------------------
frame_count = 0

# -----------------------------
# INTRUSION ZONE
# Smaller + more realistic zone
# -----------------------------
ZONE_X1, ZONE_Y1 = int(width * 0.40), int(height * 0.45)
ZONE_X2, ZONE_Y2 = int(width * 0.60), int(height * 0.80)

intrusion_count = 0
intrusion_log = []

# Store already counted IDs
counted_ids = set()

# -----------------------------
# TILING FUNCTION
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

    # -----------------------------
    # FPS CALCULATION
    # -----------------------------
    current_time = time.time()

    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0

    prev_time = current_time

    if fps > 0:
        fps_list.append(fps)

    fps_display = (
        sum(fps_list[-10:]) / len(fps_list[-10:])
        if len(fps_list) > 0 else 0
    )

    # -----------------------------
    # IMAGE ENHANCEMENT
    # -----------------------------
    frame = cv2.convertScaleAbs(
        frame,
        alpha=1.2,
        beta=10
    )

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

        # Shift detections back
        if len(det) > 0:

            det.xyxy[:, [0, 2]] += x_offset
            det.xyxy[:, [1, 3]] += y_offset

            all_detections.append(det)

    # -----------------------------
    # MERGE DETECTIONS
    # -----------------------------
    if len(all_detections) > 0:

        detections = sv.Detections.merge(all_detections)

    else:

        detections = sv.Detections.empty()

    # -----------------------------
    # TRACKING
    # -----------------------------
    detections = tracker.update_with_detections(
        detections
    )

    # -----------------------------
    # INTRUSION DETECTION
    # -----------------------------
    intrusion = False

    if detections.tracker_id is not None:

        for bbox, track_id in zip(
            detections.xyxy,
            detections.tracker_id
        ):

            x1, y1, x2, y2 = bbox

            # Center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Check if inside zone
            if (
                ZONE_X1 < cx < ZONE_X2 and
                ZONE_Y1 < cy < ZONE_Y2
            ):

                intrusion = True

                # Cleaner stable ID
                short_id = int(track_id) % 100

                # Count only once
                if short_id not in counted_ids:

                    counted_ids.add(short_id)

                    intrusion_count += 1

                    intrusion_log.append({
                        "id": short_id,
                        "frame": frame_count,
                        "time": time.time()
                    })

    # -----------------------------
    # LABELS
    # -----------------------------
    labels = []

    if detections.tracker_id is not None:

        labels = [
            f"ID {int(tid) % 100}"
            for tid in detections.tracker_id
        ]

    # -----------------------------
    # VISUALIZATION
    # -----------------------------
    annotated_frame = trace_annotator.annotate(
        frame.copy(),
        detections
    )

    annotated_frame = box_annotator.annotate(
        annotated_frame,
        detections
    )

    annotated_frame = label_annotator.annotate(
        annotated_frame,
        detections,
        labels
    )

    # -----------------------------
    # DRAW INTRUSION ZONE
    # -----------------------------
    zone_color = (0, 255, 255)

    if intrusion:
        zone_color = (0, 0, 255)

    cv2.rectangle(
        annotated_frame,
        (ZONE_X1, ZONE_Y1),
        (ZONE_X2, ZONE_Y2),
        zone_color,
        3
    )

    # -----------------------------
    # ALERT TEXT
    # -----------------------------
    if intrusion:

        cv2.putText(
            annotated_frame,
            "INTRUSION ALERT!",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # -----------------------------
    # PEOPLE COUNT
    # -----------------------------
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

    # -----------------------------
    # FPS DISPLAY
    # -----------------------------
    cv2.putText(
        annotated_frame,
        f"FPS: {fps_display:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # -----------------------------
    # FRAME NUMBER
    # -----------------------------
    cv2.putText(
        annotated_frame,
        f"Frame: {frame_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    # -----------------------------
    # INTRUSION COUNT
    # -----------------------------
    cv2.putText(
        annotated_frame,
        f"Intrusions: {intrusion_count}",
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # -----------------------------
    # FINAL RESIZE
    # -----------------------------
    annotated_frame = cv2.resize(
        annotated_frame,
        (width, height)
    )

    # -----------------------------
    # WRITE FRAME
    # -----------------------------
    out.write(annotated_frame)

    print(f"Processing {os.path.basename(image_path)}")

# -----------------------------
# RELEASE VIDEO
# -----------------------------
out.release()

# -----------------------------
# SAVE LOG FILE
# -----------------------------
with open("outputs/intrusion_log.json", "w") as f:

    json.dump(
        intrusion_log,
        f,
        indent=4
    )

print("VisDrone tracking completed!")
print(f"Total intrusions: {intrusion_count}")
print("Intrusion log saved!")