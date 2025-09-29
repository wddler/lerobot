import cv2

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(2)

# Set resolution for both cameras
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Try setting a lower framerate (e.g., 15 FPS)
cap0.set(cv2.CAP_PROP_FPS, 30)
cap1.set(cv2.CAP_PROP_FPS, 30)

# Try setting MJPEG format (check if your cameras support it via v4l2-ctl)
cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not cap0.isOpened():
    print("Error: Could not open video device 0")
    exit()
if not cap1.isOpened():
    print("Error: Could not open video device 1")
    exit()

print("Both cameras opened successfully. Press 'q' to quit.")

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0:
        print("Failed to grab frame from camera 0")
        break
    if not ret1:
        print("Failed to grab frame from camera 1")
        break

    cv2.imshow('Camera 0', frame0)
    cv2.imshow('Camera 1', frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()