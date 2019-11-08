class Queue:

  #Constructor creates a list
  def __init__(self, size,data):
      self.queue = list([data] * size)
      self.mean = 0

  #Adding elements to queue
  def enqueue(self,data):
      self.queue.insert(0,data)
      self.dequeue()
      #self.computeMean()

  #Removing the last element from the queue
  def dequeue(self):
      if len(self.queue)>0:
          return self.queue.pop()
      return ("Queue Empty!")
      
 # def computeMean(self):
      #self.mean = sum(self.queue) / len(self.queue)
