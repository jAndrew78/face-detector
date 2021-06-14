import cv2

# LOAD PRE-TRAINED DATA FROM OPENCV (HAAR CASCADE ALGO)
trained_face_data = cv2.CascadeClassifier(r'C:\Users\andyt\Desktop\source\projects\face-detector\haarcascade_frontalface_default.xml')

# CHOOSE AN IMG, CONVERT TO GREYSCALE
# img = cv2.imread(r'C:/Users/andyt/Desktop/source/projects/face-detector/face1.png')
# grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SET WEBCAM VARIABLE
webcam = cv2.VideoCapture(0)

while True:

    # READ CURRENT FRAME AND CONVERT TO GRAYSCALE
    successful_frame_read, frame = webcam.read()
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FACE DETECTION, GENERATE COORDS 
    face_coordinates = trained_face_data.detectMultiScale(grayscale_frame)

    # DRAW RECT ON ORIGINAL FRAME (NOT THE GREY ONE)
    # (x, y, w, h) = face_coordinates[0]  <-- for one face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # DISPLAY VIDEO 
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    if key==81 or key==113 or key==27:
        break

webcam.release()
print("Code Complete")
