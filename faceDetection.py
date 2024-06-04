import cv2 # Módulo OpenCV: una biblioteca de visión por computadora
import sys # Módulo Sys: acceso a variables del intérprete de Python

print("Press \"q\" to exit the program")

# Obtener la ruta del archivo de cascada Haar, en caso de no proporcionar una concreta se usa la default
cascPath = sys.argv[1] if len(sys.argv) > 1 else 'haarcascade_frontalface_default.xml'

# Crear el objeto CascadeClassifier
faceCascade = cv2.CascadeClassifier(cascPath)

# Iniciar la captura de video, el 0 hace referencia a la cámaro por defecto del sistema
video_capture = cv2.VideoCapture(0)

# Verificar si la captura de video se inició correctamente
if not video_capture.isOpened():
    print("Error: No se pudo abrir la cámara.")
    sys.exit()

while True:
    # Capturar frame por frame (ret: valor bool que indica si la captura fue exitosa)
    ret, frame = video_capture.read()
    if not ret:
        print("Error: No se pudo leer el frame.")
        break

    # Convierte el frame a escala de grises porque así la detección funciona mejor
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostros en la imagen en escala de grises
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Dibujar un rectángulo alrededor de las caras detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar el frame resultante
    cv2.imshow('Video', frame)

    # Salir del loop si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura cuando todo está hecho
video_capture.release()
cv2.destroyAllWindows()
