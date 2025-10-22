import cv2

Known_distance = 76.2   # cm
Known_width = 14.3      # cm

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)
fonts = cv2.FONT_HERSHEY_COMPLEX

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width

def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
    return (real_face_width * Focal_Length) / face_width_in_frame

def face_data(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
        return w
    return 0

ref_image = cv2.imread("reference_image.png")
ref_face_width = face_data(ref_image)
Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_face_width)
print(Focal_length_found)
cv2.imshow("ref_image", ref_image)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    face_width_in_frame = face_data(frame)

    if face_width_in_frame != 0:
        Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)
        cv2.line(frame, (30, 30), (230, 30), RED, 32)
        cv2.line(frame, (30, 30), (230, 30), BLACK, 28)
        cv2.putText(frame, f"Distance: {round(Distance, 2)} CM", (30, 35), fonts, 0.6, GREEN, 2)

    cv2.imshow("Distance Calculation", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
