from PIL import Image 
import numpy as np 
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import cv2 
import collections
import plotly.graph_objects as go
import preprocessing

digits = load_digits() 

# array to images using pillow
def convertArrayToImageUsingPIL(source):
    imageNumber = Image.fromarray(source)
    imageNumber.show()

# array to images using matplotlib
def convertArrayToImageUsingPLT(source):
    plt.figure("processing...")
    plt.gray()
    plt.imshow(source)
    plt.show()

print("database have :",len(digits.images),"pictures")
def flatMatrix(source):
    matrix = np.array(source.ravel())
    return matrix

# initialize training
# training 
imageTraining = np.array([(flatMatrix(x)) for x in digits.images[0:1790]]).astype(np.float32)
labelTraining = np.array([flatMatrix(x)   for x in digits.target[0:1790]]).astype(np.float32)

# index : Enter the handwritten digit position we need to predict
index = 1796

# Path of images 
path = "database/test_3/9.png"

# initialize matrix of handwritten words to be predicted
# testing
# predictValue = np.array([flatMatrix(digits.images[index])]).astype(np.float32)

#number predict from database
predictValue = np.array([flatMatrix(preprocessing.processing(path))]).astype(np.float32)

# initialize model
knn = cv2.ml.KNearest_create()
knn.train(imageTraining,cv2.ml.ROW_SAMPLE,labelTraining)



# print out the screen of the handwritten digits we need to predict

# convertArrayToImageUsingPLT(digits.images[index])

# print out the sreen of the handwritten digits from database
convertArrayToImageUsingPLT(preprocessing.processing(path))


# list the case of k
k = [3,107,171,303,1009,1505,1707]
listPredict = []
listAccuracy = []
print("+-------------------------------------------------+")
print("|real value : ",digits.target[index])
for i in k : 

    temp, result, nearest, distance = knn.findNearest(predictValue,i)

    print("+---------------------------------------------+")
    print("|k :",i)
    print("|predict value : ",result)
   

    listNearest = [x for x in nearest[0]]
    findAccuracy = collections.Counter(listNearest)
    # accuracy calculation
    accuracy =  round((findAccuracy[result[0][0]]/sum(findAccuracy.values()))*100,2)
    print("|nearest : ",collections.Counter(nearest[0]))
    print("|accuracy : ",accuracy,"%")
    print("+---------------------------------------------+")
    listPredict.append(result[0][0])
    listAccuracy.append(accuracy)

database = {"k" : k,"predict" : listPredict,"accuracy" : listAccuracy} 

# initialize table for case of k
def tableResult(source):
    fig = go.Figure(
        data=[go.Table(header=dict(values=['k', 'Predict Value','Accuracy Value']),
        cells=dict(values=[source["k"], source["predict"],source["accuracy"]]))
                        ])
    fig.show()
tableResult(database)
