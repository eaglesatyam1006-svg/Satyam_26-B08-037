import cv2
import numpy as np
import os
import heapq

def build_drivable_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu automatically calculates the best threshold from the image itself
    _, road_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If more than half the image got marked "road", it's probably inverted — flip it
    if np.sum(road_mask == 255) > np.sum(road_mask == 0):
        road_mask = cv2.bitwise_not(road_mask)

    kernel = np.ones((5, 5), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

    # Keep only the single largest connected white blob (the actual road)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        road_mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    return road_mask


def find_nearest_free(mask, point, max_radius=150):
    h, w = mask.shape
    x0, y0 = point
    for r in range(1, max_radius):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                    return (x, y)
    return point


def astar(mask, start, goal, step=6):
    h, w = mask.shape

    def is_free(p):
        x, y = p
        return 0 <= x < w and 0 <= y < h and mask[y, x] > 0

    def neighbors(p):
        x, y = p
        for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step),
                        (-step, -step), (-step, step), (step, -step), (step, step)]:
            n = (x + dx, y + dy)
            if is_free(n):
                yield n

    def heuristic(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    if not is_free(start):
        start = find_nearest_free(mask, start)
    if not is_free(goal):
        goal = find_nearest_free(mask, goal)

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if heuristic(current, goal) < step:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for n in neighbors(current):
            tentative = g_score[current] + heuristic(current, n)
            if n not in g_score or tentative < g_score[n]:
                g_score[n] = tentative
                came_from[n] = current
                heapq.heappush(open_set, (tentative + heuristic(n, goal), n))
    return None


def process_image(img_path, output_path, start, goal):
    img = cv2.imread(img_path)
    mask = build_drivable_mask(img)

    # Save mask image so road detection can be visually verified
    base, ext = output_path.rsplit('.', 1)
    mask_path = f"{base}_mask.{ext}"
    cv2.imwrite(mask_path, mask)

    path = astar(mask, start, goal)

    if path and len(path) > 1:
        for i in range(len(path) - 1):
            cv2.line(img, path[i], path[i + 1], (0, 255, 255), 3)
        cv2.circle(img, start, 8, (0, 255, 0), -1)   # start = green
        cv2.circle(img, goal, 8, (0, 0, 255), -1)    # goal = red
        print(f"  Path found with {len(path)} points")
    else:
        cv2.putText(img, "No valid path found", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(f"  No path found!")

    cv2.imwrite(output_path, img)


input_dir = "input"
output_dir = "output"

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # PLACEHOLDER start/goal points — adjust once mask looks correct.
        # Currently: start near the "START" arrow area (right-middle),
        # goal on the opposite side of the loop (left-middle).
        start = (int(w * 0.75), int(h * 0.55))
        goal = (int(w * 0.20), int(h * 0.55))

        print(f"Processing {filename}...")
        process_image(img_path, os.path.join(output_dir, filename), start, goal)

print("Done! Check output folder — including _mask images to verify road detection.")