import cv2 # OpenCV module: a computer vision library
import sys # Sys module: access to Python interpreter variables

print("Press \'esc\' to exit the program")

# Create the CascadeClassifier objects for the face, eyes and smile detection
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Start video capture, 0 refers to the system's default camera
video_capture = cv2.VideoCapture(0)

# Check if video capture was successfully initialized
if not video_capture.isOpened():
    print("Error: Could not open the camera.")
    sys.exit()

while True:
    # Capture frame by frame (ret: boolean value indicating if the capture was successful)
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read the frame.")
        break

    # Convert the frame to grayscale because detection works better that way
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection in the grayscale image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Draw a rectangle around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # Eyes detection in the grayscale image
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=5,
            minSize=(5, 5),
            )
        
        # Draw a rectangle around the detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        
        # Smile detection in the grayscale image
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25),
            )
        
        # Draw a rectangle around the detected smiles
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(roi_color, (xx, yy), (xx + ww, yy + hh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the loop if 'esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture when everything is done
video_capture.release()
cv2.destroyAllWindows()
