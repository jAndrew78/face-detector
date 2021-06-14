import cv2

# LOAD PRE-TRAINED DATA FROM OPENCV (HAAR CASCADE ALGO)
trained_face_data = cv2.CascadeClassifier(r'C:\Users\andyt\Desktop\source\projects\face-detector\haarcascade_frontalface_default.xml')

# CHOOSE AN IMG, CONVERT TO GREYSCALE
img = cv2.imread(r'C:/Users/andyt/Desktop/source/projects/face-detector/face1.png')
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# FACE DETECTION, GENERATE COORDS 
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

# DRAW RECT ON ORIGINAL IMG
# (x, y, w, h) = face_coordinates[0]  <-- for single face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# DISPLAY IMG 
cv2.imshow('Face Detector', img)
cv2.waitKey()


print("Code Complete")
