import cv2
import numpy as np 
import matplotlib.pyplot as plt 
# Không sử dụng với windows#############
import matplotlib                      #
import tkinter                         #
matplotlib.use('tkAgg')                #
########################################

trainData = np.random.randint(0,100,(25,2)).astype(np.float32) 
print(trainData)

label = np.random.randint(0,2,(25,1)).astype(np.float32)  # tạo random nhãn có giá trị 0 hoặc 1 

squares   = trainData[label.ravel() == 1] # hình vuông được gán nhãn  = 1 
triangles = trainData[label.ravel() == 0] # hình tam giác được gán nhãn = 0 

targetPredict = np.random.randint(0,100,(1,2)).astype(np.float32)

# vẽ đồ thị 
plt.scatter(triangles[:,0],triangles[:,1],100,'r','^') #[:,0] lấy các giá trị phía bên trái , [:,1] lấy các giá trị phía bên phải 
plt.scatter(squares[:,0],squares[:,1],100,'b','s')
plt.scatter(targetPredict[:,0],targetPredict[:,1],100,'g','o')

knn = cv2.ml.KNearest_create()
knn.train(trainData,cv2.ml.ROW_SAMPLE,label) #training data với các giá trị tương ứng 
# xét trường hợp với k = 3 
temp, result, nearest, distance = knn.findNearest(targetPredict,3)
# kết quả sau khi training
print("target to predict : ",targetPredict) # đối tượng cần gán nhãn
print('result target to predict  : ',format(result)) # kết quả sau khi đã được gán nhãn
print("nearest with : ",format(nearest)) # đối tượng gán nhãn gần với các tập dữ liệu nào nhất
print("distance is :",format(distance)) # với khoảng cách gần nhất là bao nhiêu ?
plt.show()