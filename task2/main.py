import cv2
import numpy as np
import os

def process_image(img_path, output_path):
    img = cv2.imread(img_path)
    height, width = img.shape[:2]

    # Grayscale + blur + edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Region of interest — only look at bottom half of image
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.6), int(height * 0.5)),
        (int(width * 0.4), int(height * 0.5))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    roi_edges = cv2.bitwise_and(edges, mask)

    # Detect straight lines using Hough Transform
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 50,
                             minLineLength=40, maxLineGap=100)

    left_pts, right_pts = [], []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)  # fixed: works across OpenCV versions
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.3:
                left_pts.append((x1, y1))
                left_pts.append((x2, y2))
            elif slope > 0.3:
                right_pts.append((x1, y1))
                right_pts.append((x2, y2))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # red lane lines

    # Fill drivable area between the two lanes
    if left_pts and right_pts:
        overlay = img.copy()
        pts = np.array(left_pts + right_pts[::-1], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))  # green fill
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    cv2.imwrite(output_path, img)

input_dir = "input"
output_dir = "output"

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        process_image(in_path, out_path)
        print(f"Processed: {filename}")

print("Done! Check the output folder.")