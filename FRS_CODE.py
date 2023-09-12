# IMPORTING ESSENTIAL PACKAGES FOR FACE MASK DETECTION
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# FUNCTION TO DETECT AND PREDICT FACE MASKS IN A GIVEN FRAME
def detect_and_predict_mask(frame, faceNet, maskNet):
    # OBTAINING THE DIMENSIONS OF THE FRAME
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # PASSING THE BLOB THROUGH THE NEURAL NETWORK TO DETECT FACES
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # INITIALIZE LISTS FOR FACES, THEIR LOCATIONS, AND PREDICTIONS
    faces = []
    locs = []
    preds = []

    # LOOP OVER THE DETECTIONS
    for i in range(0, detections.shape[2]):
        # EXTRACTING THE CONFIDENCE (PROBABILITY) ASSOCIATED WITH THE DETECTION
        confidence = detections[0, 0, i, 2]

        # FILTERING OUT WEAK DETECTIONS BY ENSURING CONFIDENCE IS ABOVE A THRESHOLD
        if confidence > 0.5:
            # CALCULATING THE BOUNDING BOX COORDINATES
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ENSURING THE BOUNDING BOXES ARE WITHIN THE FRAME DIMENSIONS
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # EXTRACTING, PREPROCESSING, AND ADDING THE FACE TO LISTS
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # MAKING PREDICTIONS IF AT LEAST ONE FACE WAS DETECTED
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # RETURNING FACE LOCATIONS AND THEIR MASK PREDICTIONS
    return (locs, preds)

# LOADING PRETRAINED FACE DETECTION MODEL FROM DISK
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# LOADING PRETRAINED FACE MASK DETECTION MODEL FROM DISK
maskNet = load_model("mask_detector.model")

# INITIALIZING THE VIDEO STREAM
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()

# PROCESSING FRAMES FROM THE VIDEO STREAM
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # DETECTING FACES AND PREDICTING MASKS
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # LOOPING OVER DETECTED FACE LOCATIONS AND THEIR PREDICTIONS
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # DETERMINING THE CLASS LABEL AND COLOR FOR VISUALIZATION
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # DISPLAYING LABEL AND BOUNDING BOX ON THE FRAME
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # SHOWING THE OUTPUT FRAME
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # BREAKING FROM THE LOOP IF 'q' KEY IS PRESSED
    if key == ord("q"):
        break

# CLEANING UP
cv2.destroyAllWindows()
vs.stop()
