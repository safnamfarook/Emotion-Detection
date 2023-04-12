from tensorflow.keras.utils import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

dete_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emo_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

facedetection = cv2.CascadeClassifier(dete_path)
emotionclassifier = load_model(emo_path, compile=False)
emo = ["angry" ,"disgust","fear", "happy", "sad", "surprised",
 "neutral"]

cv2.namedWindow('your_face')
cam = cv2.VideoCapture(0)
while True:
    frm = cam.read()[1]
    frm = imutils.resize(frm,width=300)
    g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    fc = facedetection.detectMultiScale(g,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

    cv = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frm.copy()
    if len(fc) > 0:
        fc = sorted(fc, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = fc
        r = g[fY:fY + fH, fX:fX + fW]
        r = cv2.resize(r, (64, 64))
        r = r.astype("float") / 255.0
        r = img_to_array(r)
        r = np.expand_dims(r, axis=0)
        pred = emotionclassifier.predict(r)[0]
        emotion_probability = np.max(pred)
        label = emo[pred.argmax()]
    else: continue

    for (i, (emotion, prob)) in enumerate(zip(emo, pred)):
                txt = "{}: {:.2f}%".format(emotion, prob * 100)
                w = int(prob * 300)
                cv2.rectangle(cv, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(cv, txt, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)

    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", cv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
