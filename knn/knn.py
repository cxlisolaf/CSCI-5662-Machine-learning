import numpy as np

import matplotlib.pylab as plt


#%matplotlib inline

class Numbers:
    """
    Class to store MNIST data
    """
    def __init__(self, location):

        import pickle, gzip

        # load data from file
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = pickle.load(f)
        f.close()

        # store for use later
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set



class Knearest:
    """
    kNN classifier
    """

    def __init__(self, X, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """
        
        from sklearn.neighbors import BallTree

        self._kdtree = BallTree(X)
        self._y = y
        self._k = k
        self._counts = self.label_counts()
        
    def label_counts(self):
        """
        Given the training labels, return a dictionary d where d[y] is  
        the number of times that label y appears in the training set. 
        """
        d={}
        for item in self._y:
            if item in d:
                d[item] += 1
            else:
                d[item] = 1
                
        #for key,value in d.items():
        #    print(str(k)+':'+str(v))  
        
        #easier way to do
        #unique,counts = np.unique(self._y,return_counts=True)     
        #d = dict(zip(unique,counts))
        #return dict({1:0, -1:0})

        return d

    def majority(self, neighbor_indices):
        """
        Given the indices of training examples, return the majority label. Break ties 
        by choosing the tied label that appears most often in the training data. 

        :param neighbor_indices: The indices of the k nearest neighbors
        """

        #assert len(neighbor_indices) == self._k, "Did not get k neighbor indices"
        
        max_freq = 0
        majority_lable = 0
        for item in self._counts:
            if self._counts[item] >= max_freq:
                max_freq = self._counts[item]
                majority_lable = item
                
                
        #major_lable_count = max(self._counts.values())
        #major_lable = max(self._counts, key = self._counts.get)
        
        examples = self._y[neighbor_indices]
        d = {}
        #print(neighbor_indices)
        for item in examples:
            if item in d:
                d[item] += 1
            else:
                d[item] = 1
        
        majority_example = {}
        max_example = 0
        
        for item in d:
            if d[item] > max_example:
                max_example = d[item]
                majority_example.clear()
                majority_example[item] = max_example
            elif d[item] == max_example:
                majority_example[item] = max_example
        
        if len(majority_example) > 1:
            return majority_lable
              
        for key in majority_example.keys():
            return key
             

    def classify(self, example):
        """
        Given an example, return the predicted label. 

        :param example: A representation of an example in the same
        format as a row of the training data
        """

        dist, ind = self._kdtree.query(example,k=self._k)
        return self.majority(ind.flatten())
               
        
    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a 2-dimensional
        numpy array of ints, C, where C[ii,jj] is the number of times an 
        example with true label ii was labeled as jj.

        :param test_x: test data 
        :param test_y: true test labels 
        """
        C = np.zeros((10,10), dtype=int)
        pred = []

        for i in range(len(test_x)):
            xx = self.classify([test_x[i]])            
            pred.append(xx)
            C[xx][test_y[i]] += 1

        #print(pred)
        #print(C)

        return C
            
    @staticmethod
    def accuracy(C):
        """
        Given a confusion matrix C, compute the accuracy of the underlying classifier.
        
        :param C: a confusion matrix 
        """
        
        return np.sum(C.diagonal()) / C.sum()
    
data = Numbers("../data/mnist.pklz")
#img = plt.imshow(data.train_x[0].reshape(28,28))
#print("the number of examples in training set = %d" % len(data.train_x))
#print("the number of examples in test set = %d" % len(data.test_y))
#print("the number of pixels in an image=%d" % len(data.train_x[0]))

accu = []
train_num = [800]
for item in train_num:
    train_x = data.train_x[0:item]
    train_y = data.train_y[0:item]
    sample = int(item/4)
    test_x = data.test_x[0:sample]
    test_y = data.test_y[0:sample]

    knn_model = Knearest(train_x,train_y)
    confusion_matrix = knn_model.confusion_matrix(test_x, test_y)
    accuracy = knn_model.accuracy(confusion_matrix)
    accu.append(accuracy)
    print(confusion_matrix)

#plt.plot(train_num,accu,lw=2)
#plt.show()
#print(accu)

"""
acc_k = []
k = [1,2,3,4,5,6,7,8,9]
for item in k:
    train_x = data.train_x[0:800]
    train_y = data.train_y[0:800]
    test_x = data.test_x[0:200]
    test_y = data.test_y[0:200]
    knn = Knerest(train_x,train_y,k=item)
    confusion_matrix = knn.confusion_matrix(test_x,test_y)
    accuracy = knn.accuracy(confusion_matrix)
    acc_k.append(accuracy)

plt.plot(k,acc_k)
plt.show()
"""