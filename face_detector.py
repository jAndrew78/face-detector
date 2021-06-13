import cv2

# LOAD PRE-TRAINED DATA FROM OPENCV (HAAR CASCADE ALGO)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# CHOOSE AN IMG, CONVERT TO GREYSCALE AND SHOW
img = cv2.imread(r'C:/Users/andyt/Desktop/source/projects/face-detector/face1.png')
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Face Detector', grayscale_img)
cv2.waitKey()


print("Code Complete")
