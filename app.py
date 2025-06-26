import cv2
import threading
import tkinter as tk
from PIL import Image, ImageTk

# Load Haar cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set up Tkinter window
root = tk.Tk()
root.title('Privacy Censor Bot')

# Video display panel
panel = tk.Label(root)
panel.pack()

# Blur slider
blur_scale = tk.Scale(root, from_=1, to=201, orient='horizontal',
                      label='Blur Strength', length=400)
blur_scale.set(51)
blur_scale.pack()

# Thread-safe flag to stop
stop_event = threading.Event()

def video_loop():
    cap = cv2.VideoCapture(0)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        k = blur_scale.get()
        if k % 2 == 0: k += 1
        for (x,y,w,h) in faces:
            roi = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, (k,k), 0)
        # Convert and display
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        panel.config(image=img)
        panel.image = img
    cap.release()

# Start video in background thread
threading.Thread(target=video_loop, daemon=True).start()

def on_close():
    stop_event.set()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
