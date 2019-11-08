# Example: prints statistics (threaded version).
# Do the same thing than stats.py but with a ThreadedNotifier's
# instance.
# This example illustrates the use of this class but the recommanded
# implementation is whom of stats.py
import time
import face_recognition
import cv2
import argparse
import pickle
import time
import datetime
import sys
sys.path.append('../common')
from utils.Person import Person
import utils.Queue as Queue
from utils.centroidTracker import CentroidTracker
from utils.csvCreator import CsvCreator
from utils.facerec import Facerec
import datetime

myFacerec = Facerec()


if __name__ == '__main__':

    #myFacerec.updateData()
    #load pickle and nameList
    nameList = myFacerec.getData()
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("rtsp://192.168.0.219:554/media1.sdp")
    #cap = cv2.VideoCapture("videos/videoprova.mp4")
    #cap = cv2.VideoCapture("videos/videoLuca.mp4")
    #cap = cv2.VideoCapture("videos/videoTracking.mp4")
    time.sleep(2.0)
    #initialize csv
    #csv=CsvCreator()
    #csv.createCsv()

    while True:
        
        start = time.time()
        names = []
        ret, frame = cap.read()
        face_locations = myFacerec.doFaceDet(frame)
        namesEnter, frame_plot = myFacerec.doFaceRec(face_locations, frame, nameList)
        end = time.time()
        elapsed = end - start
        fps = 1 / elapsed
        
        cv2.putText(frame_plot, str(fps), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255) ,2)
        cv2.imshow("window", frame_plot)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

