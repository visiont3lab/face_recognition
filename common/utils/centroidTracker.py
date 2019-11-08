import cv2
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid,name,idObj):
        self.objects[self.nextObjectID] = {"centroid":centroid, "state":True,"name":name,"idObj":idObj,"size":0}
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        self.objects[objectID]["state"]=False


    def deleteObject(self,objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        self.nextObjectID -= 1

    def changeName(self,idObj,name):
        self.objects[idObj]["name"]=name

    def changeSize(self,idObj,size):
        self.objects[idObj]["size"]=size


    def update(self, rects,names,personDict):
        maxDistance = 100
        # if there are not objects in current frame
        # then increase disappeared counter
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # find centroid of new objects
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX =  int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                for key, person in personDict.items():
                    if(person.found==True and person.name==names[i]):
                        self.register(inputCentroids[i],names[i],self.nextObjectID)

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [x["centroid"] for x in self.objects.values()]

            # calculate distance between centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # find the smallest distances
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]

                # check distance between old and new object
                # if distance < threshold value, then update its position
                if D[row,col] < maxDistance:
                    self.objects[objectID]["centroid"] = inputCentroids[col]
                    self.disappeared[objectID] = 0
                    usedRows.add(row)
                    usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # some objects disappeared
            for row in unusedRows:
                objectID = objectIDs[row]

                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # new objects are registered
            for col in unusedCols:
                    for key, person in personDict.items():
                            if(person.found==True and person.name==names[i]):
                                self.register(inputCentroids[col],names[i],self.nextObjectID)

        return self.objects
