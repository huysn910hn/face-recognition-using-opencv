import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join, splitext
import sys
cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
path_train = '/xla_python/extract'
file = [f for f in listdir(path_train) if isfile(join(path_train, f))] # Liệt kê tất cả các file trong thư mục
names = []
training_data, labels = [], []
for i, filename in enumerate(file):
    train_image_path = join(path_train, filename)
    name, _ = splitext(filename)  # Tách tên file và phần mở rộng
    image = cv.imread(train_image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print(f"No images available: {train_image_path}")
    else:
        training_data.append(np.asarray(image, dtype=np.uint8))
        labels.append(i)  # Sử dụng chỉ số file làm nhãn
        names.append(name) # Lưu tên tương ứng với nhãn
def detect(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gau = cv.GaussianBlur(img_gray, (5,5), 0)
    faces = cascade.detectMultiScale(img_gau, 1.1, 5)
    if len(faces) == 0:
        return None, img
    else:
        for (x, y, w, h) in faces:
            roi = img_gray[y:y + h, x:x + w]
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return roi, img
distance = []
def processing_face():
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
    
        if not ret:
            print("Error")
            break

        face, image = detect(frame)
        try:
            if face is not None:
                res = model.predict(face) # res[0] là nhãn, res[1] là khoảng cách dự đoán
                if res[1] < 400:
                    confidence = float("{0:.2f}".format((100 * (1 - (res[1]) / 300))))
                    if confidence > 80:
                        recognized_name = names[res[0]]
                        dis = str(confidence) + "% similar to " + recognized_name
                        cv.putText(image, recognized_name, (250, 450), cv.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2)
                        cv.putText(image, dis, (10, 30), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                    else:
                        cv.putText(image, "Unknown", (250, 450), cv.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
                else:
                    cv.putText(image, "Cannot detect face", (250, 450), cv.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
            else:
                cv.putText(image, "Cannot found face", (250, 450), cv.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 2)
            cv.imshow('face recognition', image)
        except Exception as e:
            print(f"Error: {e}")
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__": 
    if len(training_data) > 0:
        labels = np.asarray(labels, dtype=np.int32)
        training_data = np.asarray(training_data, dtype=np.uint8)
        model = cv.face.LBPHFaceRecognizer_create()
        model.train(training_data, labels)
        print("Model training complete")
        processing_face()
    else:
        print("No training data found. Please check the image files.")
        sys.exit(1)