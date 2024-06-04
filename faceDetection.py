import cv2 # OpenCV module: a computer vision library
import sys # Sys module: access to Python interpreter variables

print("Press \'esc\' to exit the program")

# Get the Haar cascade file path, use the default if none is provided
cascPath = sys.argv[1] if len(sys.argv) > 1 else 'haarcascade_frontalface_default.xml'

# Create the CascadeClassifier object
faceCascade = cv2.CascadeClassifier(cascPath)

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

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the loop if 'esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture when everything is done
video_capture.release()
cv2.destroyAllWindows()
