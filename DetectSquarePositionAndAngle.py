import cv2
import numpy as np


def detect_squares(image_path):
  # 1. Load the image
  img = cv2.imread(image_path)
  if img is None:
    print(f"Error: Could not load image from {image_path}")
    return

  # 2. Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # 3. Threshold: Background and gaps are white (~255), squares are grey (~191).
  # THRESH_BINARY_INV turns grey squares into white (255) and white background into black (0).
  _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

  # Clean up any tiny artifacts with morphological closing
  kernel = np.ones((3, 3), np.uint8)
  thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

  # 4. Find external contours of the isolated squares
  contours, _ = cv2.findContours(
      thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  square_count = 0
  print(f"{'='*30}\nDetected Squares Analysis:\n{'='*30}")

  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 500:  # Filter out small noise
      continue

    # Get the minimum area rotated bounding rectangle
    rect = cv2.minAreaRect(cnt)
    center, (width, height), _ = rect

    if width < 10 or height < 10:
      continue

    # Check aspect ratio (width vs height should be close to 1.0 for a square)
    aspect_ratio = float(width) / height
    if not (0.75 <= aspect_ratio <= 1.25):
      continue

    # Check fill ratio (Contour Area / Bounding Box Area)
    # A solid square should have a fill ratio close to 1.0
    rect_area = width * height
    fill_ratio = area / rect_area
    if not (0.75 <= fill_ratio <= 1.05):
      continue

    square_count += 1

    # Calculate precise tilt angle from the bounding box edge vector
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    edge_vector = box[1] - box[0]
    angle = np.arctan2(edge_vector[1], edge_vector[0]) * 180.0 / np.pi

    # Normalize angle to range [-45, 45]
    while angle < -45:
      angle += 90
    while angle > 45:
      angle -= 90

    print(f"Square #{square_count}:")
    print(f"  - Center (x, y): ({center[0]:.1f}, {center[1]:.1f})")
    print(f"  - Tilt Angle:    {angle:.1f}°")
    print(f"  - Size:          {width:.1f} x {height:.1f}\n")

    # Draw the rotated bounding box on the original image
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    # Mark the center point
    cv2.circle(img, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)

    # Put label on the image
    cv2.putText(
        img,
        f"#{square_count} ({angle:.1f}deg)",
        (int(center[0]) - 35, int(center[1]) - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2,
    )

  print(f"Total squares detected: {square_count}")

  # 5. Display the result
  cv2.imshow("Detected Squares", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


# Example usage:
detect_squares(
    "D:/self/practice/localRepo/squares in squares/Squares-in-Squares/images/5.png"
)