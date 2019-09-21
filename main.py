from PIL import Image 
import numpy as np 
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import cv2 
import collections
import plotly.graph_objects as go

digits = load_digits() 

# ARRAY TO IMAGE USING PIL
def convertArrayToImageUsingPIL(source):
    imageNumber = Image.fromarray(source)
    imageNumber.show()

# ARRAY TO IMAGE USING SCIKIT LEARN  + MATPLOTLIB 
def convertArrayToImageUsingPLT(source):
    plt.gray()
    plt.imshow(source)
    plt.show()

print("database have :",len(digits.images),"pictures")
def flatMatrix(source):
    matrix = np.array(source.ravel())
    return matrix

# khởi tạo bộ dữ liệu để training 
# training 

imageTraining = np.array([(flatMatrix(x)) for x in digits.images[0:1790]]).astype(np.float32)
labelTraining = np.array([flatMatrix(x)   for x in digits.target[0:1790]]).astype(np.float32)

# index : nhập vị trí chữ số viết tay mà chúng ta cần dự đoán

index = 1796

# khởi tạo ma trận chữ số viết tay cần dự đoán 
# testing

predictValue = np.array([flatMatrix(digits.images[index])]).astype(np.float32)

# khởi tạo model

knn = cv2.ml.KNearest_create()
knn.train(imageTraining,cv2.ml.ROW_SAMPLE,labelTraining)



# in ra màn hình chữ số viết tay mà chúng ta cần dự đoán 
convertArrayToImageUsingPLT(digits.images[index])

# liệt kê các trường hợp của k 
k = [3,107,171,303,1009,1505,1707]
listPredict = []
listAccuracy = []
print("|real value : ",digits.target[index])
for i in k : 

    temp, result, nearest, distance = knn.findNearest(predictValue,i)

    print("+---------------------------------------------+")
    print("|k :",i)
    print("|predict value : ",result)
   

    listNearest = [x for x in nearest[0]]
    findAccuracy = collections.Counter(listNearest)
    # tính toán độ chính xác 
    accuracy =  round((findAccuracy[result[0][0]]/sum(findAccuracy.values()))*100,2)
    print("|accuracy : ",accuracy,"%")
    print("+---------------------------------------------+")
    listPredict.append(result[0][0])
    listAccuracy.append(accuracy)

database = {"k" : k,"predict" : listPredict,"accuracy" : listAccuracy} 

def tableResult(source):
    fig = go.Figure(
        data=[go.Table(header=dict(values=['k', 'Predict Value','Accuracy Value']),
        cells=dict(values=[source["k"], source["predict"],source["accuracy"]]))
                        ])
    fig.show()
tableResult(database)