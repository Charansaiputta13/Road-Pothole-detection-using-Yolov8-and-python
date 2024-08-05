import cv2
import numpy as np
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')
class_names = model.names

# Open the video file
cap = cv2.VideoCapture('p.mp4')
count = 0
frame_interval = 10  # Process every 3rd frame

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    if count % frame_interval != 0:
        continue
    
    img = cv2.resize(img, (1024, 500))
    h, w, _ = img.shape
    results = model.predict(img)

    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs

        if masks is not None:
            masks = masks.data.cpu().numpy()  # Ensure masks are NumPy arrays
            for seg, box in zip(masks, boxes):
                seg_resized = cv2.resize(seg, (w, h))

                contours, _ = cv2.findContours((seg_resized).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, width, height = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                 
    cv2.imshow('Video', img)  # Display the image using OpenCV
    if cv2.waitKey(10) & 0xFF == ord('q'):  # Wait for 10ms to make the video smoother
        break

cap.release()
cv2.destroyAllWindows()
