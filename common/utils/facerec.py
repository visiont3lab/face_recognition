import os
import face_recognition
import cv2
import argparse
import pickle
import time
import datetime
import numpy as np
from utils.Person import Person
import utils.Queue as Queue
from utils.centroidTracker import CentroidTracker
from imutils import paths

class Facerec():
    def __init__(self):
        self.modelType = "cnn"  #hog
        self.number_of_times_to_upsample = 0
        self.yTopCrop=0
        self.yBottomCrop=800
        self.xLeftCrop=300
        self.xRightCrop=1200
        #disable found name person
        self.disableFoundName=100
        # initialize our centroid tracker and frame dimensions
        self.centroidList= []
        self.tracker = CentroidTracker()
        self.countFrame=0
        self.personList=[]
        self.personDict={}

    def setup_frame(self, frame):
        # prepare the frame and cutting
        #frame = cv2.resize(frame, (1280, 720), interpolation = cv2.INTER_CUBIC)
        frame = frame[self.yTopCrop:self.yBottomCrop, self.xLeftCrop:self.xRightCrop]
        return frame

    def updateData(self):
        print("Update encodings")
        imagePaths = list(paths.list_images("../common/faces"))
        #print("People in the list are: " + str(len(imagePaths)))
        knownEncodings = []
        knownNames = []
        for (i, imagePath) in enumerate(imagePaths):
            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb,model=self.modelType)
            encodings = face_recognition.face_encodings(rgb, boxes)
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)
                print("Processing  Encodings " +  name)
        print("Writing to pickle file.")
        self.data = {"encodings": knownEncodings, "names": knownNames}
        f = open("../common/models/encodings.pickle", "wb")
        f.write(pickle.dumps(self.data))
        f.close

    def getData(self):
        print("Load encodings from pickle.")
        nameList = []
        time.sleep(5)
        self.data = pickle.loads(open("../common/models/encodings.pickle", "rb").read())
        for datasetName in self.data["names"]:
            if len(self.personList) == 0 or (self.personList[-1].name != datasetName ):
                nameList.append(datasetName)
                self.personList.append(Person(datasetName))
                self.personDict.update({datasetName:Person(datasetName)})
                print("Initializing Persons list=", end="")
                print(datasetName)
        return nameList
    '''
    Do face recognition and write the names over the image. Return a list of tuples compose by name and locations
    @param rects: the locations of the faces
    @param frame: the rgb image
    @return a list of tuples
    '''
    def doFaceRec(self,rects, frame, nameList):
        namesEnter=[]
        frame = self.setup_frame(frame)
        boxes = [(top, right, bottom, left) for (top,right,bottom,left) in rects]
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []
        rectTracker=[]
        for index,encoding in enumerate(encodings):
            matches = face_recognition.compare_faces(self.data["encodings"], encoding, tolerance=0.5)
            #print(matches)
            #print(face_recognition.face_distance(data["encodings"], encoding))
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, boolVar) in enumerate(matches) if boolVar]
                counts = {}
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            # update queue every person if is present
            for key, person in self.personDict.items():
                #disable detection of specific person for TOT frame
                if  person.lastFoundFrame + self.disableFoundName < self.countFrame:
                    person.logging = True
                if person.name == name and person.logging == True:
                       person.addQueueValue(1)
                       #print("viewed", name)

            names.append(name)

        # Add 0 to not viewed persons
        for item in nameList:
            if item not in names:
                for key, person in self.personDict.items():
                    if(person.name == item):
                        # print("Person not found ", item)
                        person.addQueueValue(0)
                        person.addCenter((0,0))
                        person.size=0

        # Draw text and rectangles of the recognition and generate rect for tracked update, Have different order of top,bottom,left,right
        for ((top,right,bottom,left), name) in zip(boxes, names):
            y = top - 15 if top -15 > 15 else top + 15
            cv2.putText(frame, name, (left,top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 2)
            rectTracker.append((left,top,right,bottom))
            #print("_positionX_",left+(right - left)/2,"_positionY_",top+(bottom - top)/2,"_size_",bottom-top,"_at_Frame_",self.countFrame)

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        #Tracking
        objectsTracked = self.tracker.update(rectTracker,names,self.personDict)

        #check if a person is present more than 4 times over 8. If yes became found=true
        for key, person in self.personDict.items():
            person.checkQueue()

        #loop on object tracked and check if update the name of object tracked
        for index,(objectID, object) in enumerate(list(objectsTracked.items())):
            centroidTracked = object["centroid"]
            nameTracked=object["name"]
            idTracked=object["idObj"]
            sizeTracked=object["size"]
            cv2.putText(frame, "{}".format(idTracked+1), (centroidTracked[0]-10, centroidTracked[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, "{}".format(nameTracked), (centroidTracked[0]-50, centroidTracked[1]-50),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.circle(frame, (centroidTracked[0], centroidTracked[1]), 4, (0, 255, 255), -1)
            #loop sui rettangoli della recognition con i nomi associati, devo fare questo loop perchè non sono stessi rettangoli del tracking
            for indexDetection,((top,right,bottom,left), name) in enumerate(zip(boxes, names)):
                    #controllo che il centroid sia associato al corretto rettangolo con il corretto name trovato
                    if(centroidTracked[1]>top and centroidTracked[1]<bottom and centroidTracked[0]>left and centroidTracked[0]<right):
                        #faccio l aggiornamento del nome solo se quella persona è stata trovata ed il logging è abilitato per quella persona.
                            if(names[indexDetection]!="Unknown" and self.personDict[names[indexDetection]].found==True):
                                    #print("changeName at frame",self.countFrame,"with name",names[indexDetection] )
                                    self.tracker.changeName(idTracked,names[indexDetection])
                                    self.personDict[names[indexDetection]].cleanQueue()
                            #aggiorno la posizione,size della persona basandomi sul tracking della persona non sul recognition
                            self.tracker.changeSize(idTracked,bottom-top)
                            print("People tracked:",nameTracked,"at _positionX_",centroidTracked[0],"_positionY_",centroidTracked[1],"_size_",bottom-top,"_frame_",self.countFrame)

        #check if that tracked object is disappeared at position of entering
            if(self.tracker.objects[idTracked]["state"]==False):
                if(self.tracker.objects[idTracked]["size"]>50 and centroidTracked[1]<380):
                    namesEnter.append(nameTracked)
                    print("ENTER -----------------People:",nameTracked,"at _positionX_",centroidTracked[0],"_positionY_",centroidTracked[1],"_size_",self.tracker.objects[idTracked]["size"],str(datetime.datetime.now().replace(microsecond=0)))
                self.tracker.deleteObject(idTracked)

        return namesEnter, frame

    '''
    Do face detection and return a list of tuples composed by top, right, bottom, left of the rectangles
    @param gray: the whole image from the camera in grayscale format
    @param type: either cnn or hog
    @return a list of tuples
    '''
    def doFaceDet(self,frame):
        self.countFrame+=1
        frame = self.setup_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(gray, number_of_times_to_upsample=self.number_of_times_to_upsample, model=self.modelType)
        return face_locations

    def resize_face_locations(self, face_locations, input_frame_width, input_frame_height):
        # Update face_locations --> we want them with respect tot the input frame not process_frame
        for i in range(0,len(face_locations)):
            top, right, bottom, left = face_locations[i]
            #print(top,right,bottom,left)
            process_frame_height=360
            process_frame_width=640
            blank_image = np.zeros((process_frame_height,process_frame_width), np.uint8)
            cv2.rectangle(blank_image, (left,top), (right,bottom), (255), 2)
            blank_image = cv2.resize(blank_image, (input_frame_width, input_frame_height), interpolation = cv2.INTER_CUBIC)
            # find contours
            ctr, _ = cv2.findContours(blank_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #cv2.drawContours(input_frame,ctr,-1,(0,255,0),3);
            #cv2.imshow("i",input_frame)
            #cv2.waitKey(0)
            x, y, w, h = cv2.boundingRect(ctr[0]) # We have always one contour
            #print(top,right,bottom,left)
            face_locations[i] = (y,x+w,y+h,x) #(top,right,bottom, left)
            #cv2.rectangle(input_frame, (x,y), (x+w,y+h), (255), 2)
            #cv2.rectangle(input_frame, (left,top), (right,bottom), (255), 2)
            #cv2.imshow("i",input_frame)
            #cv2.waitKey(0)
        # Face locations are given wiht respect to the input_frame
        return face_locations

    def doFaceDet_simple(self,input_frame):
        # Return the face locations with respect process frame
        process_frame = input_frame #cv2.resize(input_frame, (self.process_frame_width, self.process_frame_height), interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(gray, number_of_times_to_upsample=self.number_of_times_to_upsample, model=self.modelType)
        return face_locations, process_frame

    def doFaceRec_simple(self,rects, frame, nameList):
        namesEnter=[]
        boxes = [(top, right, bottom, left) for (top,right,bottom,left) in rects]
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []
        rectTracker=[]
        for index,encoding in enumerate(encodings):
            matches = face_recognition.compare_faces(self.data["encodings"], encoding, tolerance=0.5)
            #print(matches)
            #print(face_recognition.face_distance(data["encodings"], encoding))
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, boolVar) in enumerate(matches) if boolVar]
                counts = {}
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            # update queue every person if is present
            for key, person in self.personDict.items():
                #disable detection of specific person for TOT frame
                if  person.lastFoundFrame + self.disableFoundName < self.countFrame:
                    person.logging = True
                if person.name == name and person.logging == True:
                       person.addQueueValue(1)
                       #print("viewed", name)

            names.append(name)

        # Add 0 to not viewed persons
        for item in nameList:
            if item not in names:
                for key, person in self.personDict.items():
                    if(person.name == item):
                        # print("Person not found ", item)
                        person.addQueueValue(0)
                        person.addCenter((0,0))
                        person.size=0

        # Draw text and rectangles of the recognition and generate rect for tracked update, Have different order of top,bottom,left,right
        for ((top,right,bottom,left), name) in zip(boxes, names):
            y = top - 15 if top -15 > 15 else top + 15
            cv2.putText(frame, name, (left,top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 2)
            rectTracker.append((left,top,right,bottom))
            #print("_positionX_",left+(right - left)/2,"_positionY_",top+(bottom - top)/2,"_size_",bottom-top,"_at_Frame_",self.countFrame)

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        #Tracking
        objectsTracked = self.tracker.update(rectTracker,names,self.personDict)

        #check if a person is present more than 4 times over 8. If yes became found=true
        for key, person in self.personDict.items():
            person.checkQueue()

        #loop on object tracked and check if update the name of object tracked
        for index,(objectID, object) in enumerate(list(objectsTracked.items())):
            centroidTracked = object["centroid"]
            nameTracked=object["name"]
            idTracked=object["idObj"]
            sizeTracked=object["size"]
            cv2.putText(frame, "{}".format(idTracked+1), (centroidTracked[0]-10, centroidTracked[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(frame, "{}".format(nameTracked), (centroidTracked[0]-50, centroidTracked[1]-50),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.circle(frame, (centroidTracked[0], centroidTracked[1]), 4, (0, 255, 255), -1)
            #loop sui rettangoli della recognition con i nomi associati, devo fare questo loop perchè non sono stessi rettangoli del tracking
            for indexDetection,((top,right,bottom,left), name) in enumerate(zip(boxes, names)):
                    #controllo che il centroid sia associato al corretto rettangolo con il corretto name trovato
                    if(centroidTracked[1]>top and centroidTracked[1]<bottom and centroidTracked[0]>left and centroidTracked[0]<right):
                        #faccio l aggiornamento del nome solo se quella persona è stata trovata ed il logging è abilitato per quella persona.
                            if(names[indexDetection]!="Unknown" and self.personDict[names[indexDetection]].found==True):
                                    #print("changeName at frame",self.countFrame,"with name",names[indexDetection] )
                                    self.tracker.changeName(idTracked,names[indexDetection])
                                    self.personDict[names[indexDetection]].cleanQueue()
                            #aggiorno la posizione,size della persona basandomi sul tracking della persona non sul recognition
                            self.tracker.changeSize(idTracked,bottom-top)
                            print("People tracked:",nameTracked,"at _positionX_",centroidTracked[0],"_positionY_",centroidTracked[1],"_size_",bottom-top,"_frame_",self.countFrame)

        #check if that tracked object is disappeared at position of entering
            if(self.tracker.objects[idTracked]["state"]==False):
                if(self.tracker.objects[idTracked]["size"]>100 and centroidTracked[1]<380):
                    namesEnter.append(nameTracked)
                    print("ENTER -----------------People:",nameTracked,"at _positionX_",centroidTracked[0],"_positionY_",centroidTracked[1],"_size_",self.tracker.objects[idTracked]["size"],str(datetime.datetime.now().replace(microsecond=0)))
                self.tracker.deleteObject(idTracked)

        return namesEnter, frame
