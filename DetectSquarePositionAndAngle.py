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

  # 3. Apply Gaussian blur to reduce noise
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)

  # 4. Use Adaptive Thresholding to handle varying lighting conditions
  thresh = cv2.adaptiveThreshold(
      blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
  )

  # 5. Find contours in the binary image
  contours, _ = cv2.findContours(
      thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )

  square_count = 0
  print(f"{'='*30}\nDetected Squares Analysis:\n{'='*30}")

  for cnt in contours:
    # Filter out small noise contours by area (adjust threshold based on your image resolution)
    area = cv2.contourArea(cnt)
    if area < 200:
      continue

    # Approximate the contour to a polygon
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

    # A square should have 4 vertices and form a convex polygon
    if len(approx) == 4 and cv2.isContourConvex(approx):
      # Calculate the minimum area rotated bounding rectangle
      rect = cv2.minAreaRect(cnt)
      center, (width, height), angle = rect

      # Avoid division by zero
      if width > 0 and height > 0:
        # Check aspect ratio to ensure it's a square (width/height close to 1.0)
        aspect_ratio = float(width) / height
        if 0.85 <= aspect_ratio <= 1.15:
          square_count += 1

          # Note on OpenCV angle: minAreaRect returns angles between -90 and 0 degrees.
          # If width < height, OpenCV might report width and height swapped along with a 90-degree offset.
          if width < height:
            angle += 90

          print(f"Square #{square_count}:")
          print(f"  - Center (x, y): ({center[0]:.1f}, {center[1]:.1f})")
          print(f"  - Tilt Angle:    {angle:.1f}°")
          print(f"  - Size:          {width:.1f} x {height:.1f}\n")

          # Draw the rotated bounding box on the original image
          box = cv2.boxPoints(rect)
          box = np.intp(box)
          cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

          # Mark the center point
          cv2.circle(img, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)

          # Put label on the image
          cv2.putText(
              img,
              f"#{square_count} ({angle:.1f}deg)",
              (int(center[0]) - 30, int(center[1]) - 15),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              (255, 0, 0),
              2,
          )

  print(f"Total squares detected: {square_count}")

  # 6. Display the result
  cv2.imshow("Detected Squares", img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


# Example usage:
# detect_squares("path_to_your_image.jpg")