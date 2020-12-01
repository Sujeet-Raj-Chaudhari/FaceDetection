import cv2
from random import randrange

#Load some pre-trained data on face frontals(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect faces input
#img = cv2.imread('RDJ.png')
webcam = cv2.VideoCapture(0)

##### Iterate forever for frames

while True:
  ####Read current forme
  successful_frame_read, frame = webcam.read()

  #Must convert to grayscale
  grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  #Detect faces
  face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

  #Create  Rectangle for face
  for (x, y, w, h) in face_coordinates:
      cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 10)
  
  cv2.imshow('Sujeet Face detetctor', frame)
  key = cv2.waitKey(1)

  if key == 81 or key == 113:
    break

