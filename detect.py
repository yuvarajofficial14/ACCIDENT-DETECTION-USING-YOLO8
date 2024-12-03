import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Open the video file or webcam
cap = cv2.VideoCapture("D:\\p\\1\\data\\testing.mp4")  # Use 0 for webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Visualize results
    frame = results[0].plot()

    # Display the frame
    cv2.imshow("Accident Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
