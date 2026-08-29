import cv2
import numpy as np

img = cv2.imread("input/1.jpeg")  # change filename if needed
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Print HSV value at a few sample points on the road (eyeball coordinates)
# Click around your image mentally: pick a pixel you're sure is ON the gray road
sample_points = [(400, 300), (600, 400), (300, 600)]  # adjust based on image size

h, w = img.shape[:2]
print(f"Image size: {w}x{h}")

for x, y in sample_points:
    if x < w and y < h:
        print(f"Pixel ({x},{y}) HSV: {hsv[y, x]}")

# Also print overall HSV stats to understand the range
print("\nSaturation channel stats:")
print(f"Min: {hsv[:,:,1].min()}, Max: {hsv[:,:,1].max()}, Mean: {hsv[:,:,1].mean():.1f}")
print("\nValue (brightness) channel stats:")
print(f"Min: {hsv[:,:,2].min()}, Max: {hsv[:,:,2].max()}, Mean: {hsv[:,:,2].mean():.1f}")