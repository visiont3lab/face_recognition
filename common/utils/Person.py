import utils.Queue as Queue

queue_size = 16

class Person:
    def __init__(self, name):
        self.name = name
        self.centerList = Queue.Queue(queue_size,(0,0))
        self.size = 0
        self._foundQueue = Queue.Queue(queue_size,0)
        self.found = False
        self.lastFoundFrame = 0
        self.logging = True
        self.idCentroid=0
        #persona che è stata vista più di 4 volte e che quindi può entrare
        self.possibleEntering=False
    def checkQueue(self):
        # Implement a better counter, when removing an element update the counter frome the previous value
        counter = 0
        for element in iter (self._foundQueue.queue):
            if element == 1:
              counter += 1
        if counter >= 8:
            self.found = True
        else:
            self.found = False

    def addQueueValue(self, val):
        self._foundQueue.enqueue(val)

    def cleanQueue(self):
        self._foundQueue = Queue.Queue(queue_size,0)
        self.centerList=Queue.Queue(queue_size,(0,0))
        self.size=0

    def addCenter(self,val):
        self.centerList.enqueue(val)

    def checkEntering(self):
        maxX=max(self.centerList.queue,key=lambda x:x[0])[0]
        minX=min(self.centerList.queue,key=lambda x:x[0])[0]
        if(self.size>90 and maxX>230):
            #print(self.size,"_",maxX)
            return True
        else:
            return False
