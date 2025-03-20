import numpy as np
import cv2 as cv
import sys
import time

cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

def extract(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gau = cv.GaussianBlur(img_gray, (5, 5), 0)
    faces = cascade.detectMultiScale(img_gau, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, img
    else:
        for (x, y, w, h) in faces:
            crop = img[y:y+h, x:x+w]
            break
        return crop, img

def processing_img(name):
    cap = cv.VideoCapture(0)
    count = 0 
    count_not_face = 0
    while (True and count < 300):
        ret, frame = cap.read()
        if not ret:
            print("Camera not found")
            break
        cropped_face, _ = extract(frame)
        if cropped_face is not None:
            count += 1
            face = cv.resize(cropped_face, (500, 500))
            face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
            path = f'extract/{name}_{count}.jpg'
            cv.imwrite(path, face)
            name_text = f'{name}_{count}'
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(face, name_text, (50, 50), font, 1, (0,255,0), 2)
            cv.imshow('processing', face)
        else:
            print("Face not found")
            count_not_face += 1
            time.sleep(1)
            if count_not_face >= 10:
                print("Face not found, Please adjust the camera")
                break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    print("sampling completed")
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1] 
        processing_img(name)