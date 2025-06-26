import cv2
import numpy as np

# 1. Load the face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 2. Initialize webcam
cap = cv2.VideoCapture(0)

# 3. Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam. Make sure it's connected and not in use.")
    exit()

# 4. Process frames in a loop
while True:
    ret, frame = cap.read()   # ret = success flag, frame = current image
    if not ret:
        print("Failed to grab frame. Exiting…")
        break

    # 5. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 6. Detect faces: returns list of rectangles (x, y, w, h)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,    # image size reduction per scale
        minNeighbors=5,     # how many neighbors each candidate rect should have
        minSize=(30, 30)    # smallest face size to detect
    )

    # 7. Loop over detected faces and blur each
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y+h, x:x+w]

        # Apply a blur filter
        blurred_face = cv2.GaussianBlur(face_roi, (101, 101), 0)

        # Put the blurred face back into the frame
        frame[y:y+h, x:x+w] = blurred_face

    # 8. Display the result
    cv2.imshow("Privacy Censor Bot 🤖", frame)

    # 9. Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting… Goodbye!")
        break

# 10. Clean up
cap.release()
cv2.destroyAllWindows()
