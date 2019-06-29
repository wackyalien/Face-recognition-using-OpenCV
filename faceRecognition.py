

import cv2
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
id = raw_input('enter id')
sample = 0
while True:
    ret , img = cam.read()
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    face = facedetect.detectMultiScale(gray , 1.3 , 5)
    for (x,y,w,h) in face:
        sample = sample + 1
        cv2.imwrite('dataset/user' +'.'+ str(id) + '.' + str(sample) +'.jpg', gray[y:y+h , x:x+w])
        cv2.rectangle(img , (x,y) , (x+w , y+h) , (0,0,255) , (2))
        cv2.waitKey(100)
    cv2.imshow('faces' , img)
    cv2.waitKey(1)
    if (sample>20):
        break
cam.release()
cv2.destroyAllWindows()
