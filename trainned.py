import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'

def getImagesWithID(path):
    imagepaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagepath in imagepaths:
        faceImg = Image.open(imagepath).convert('L')
        facenp = np.array(faceImg , 'uint8')
        ID = int(os.path.split(imagepath)[-1].split('.')[1])
        faces.append(facenp)
        print ID
        IDs.append(ID)
        cv2.imshow('training' , facenp)
        cv2.waitKey(10)
    return IDs , faces

Ids , faces = getImagesWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.save('recognizer/trainningData.yml')
cv2.destroyAllWindows()
 
