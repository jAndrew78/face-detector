import cv2

# LOAD PRE-TRAINED DATA FROM OPENCV (HAAR CASCADE ALGO)
trained_face_data = cv2.CascadeClassifier(r'C:\Users\andyt\Desktop\source\projects\face-detector\haarcascade_frontalface_default.xml')

# CHOOSE AN IMG, CONVERT TO GREYSCALE
img = cv2.imread(r'C:/Users/andyt/Desktop/source/projects/face-detector/face1.png')
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# DETECT FACES, GENERATE COORDS AND DRAW RECT
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
cv2.rectangle(img, (94, 38), (330+94, 330+38), (0, 255, 0), 2)



cv2.imshow('Face Detector', img)
cv2.waitKey()


print("Code Complete")
