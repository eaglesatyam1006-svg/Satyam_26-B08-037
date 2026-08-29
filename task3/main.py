import cv2
import numpy as np
import os
import json

def detect_obstacles(img_path, output_path, coords_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold for white/bright blobs (potholes described as white circular blobs)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours (blobs)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:  # skip tiny noise specks
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        count += 1
        detections.append({
            "id": count,
            "x": x, "y": y,
            "width": w, "height": h
        })

    # Overlay total count as text
    cv2.putText(img, f"Total detected: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)

    with open(coords_path, "w") as f:
        json.dump(detections, f, indent=2)

    return count

input_dir = "input"
output_dir = "output"

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        coords_path = os.path.join(output_dir, filename.rsplit('.', 1)[0] + "_coords.json")

        n = detect_obstacles(in_path, out_path, coords_path)
        print(f"{filename}: {n} obstacles/potholes detected")

print("Done! Check the output folder.")