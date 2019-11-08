import os
import cv2
import face_recognition
import argparse
import pickle
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-i", required=False, default="faces", help="path to input directory of images.")
ap.add_argument("-e", required=False, default="models/encodings.pickle", help="path to encodings")
ap.add_argument("-d", type=str, default="cnn", help="detection method: cnn or hog")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["i"]))
print("People in the list are: " + str(len(imagePaths)))

knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    #print("Processing " +  name)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #boxes = face_recognition.face_locations(rgb, model=args["d"])

		# detect the (x, y)-coordinates of the bounding boxes
		# corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb,model=args["d"])


    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
        print("Processing  Encodings " +  name)

print("Writing to pickle file.")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["e"], "wb")
f.write(pickle.dumps(data))
f.close
