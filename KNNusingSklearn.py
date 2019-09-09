
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import neighbors
datatrainning=np.random.randint(0,100,(25,2)).astype(np.float32)
label=np.random.randint(0,2,(25,1)).astype(np.float32)
square=datatrainning[label.ravel()==1]
triangles=datatrainning[label.ravel()==0]
randomPlot=np.random.randint(0,100,(1,2)).astype(np.float32)
plt.plot(square[:,0],square[:,1],"rs")
plt.plot(triangles[:,0],triangles[:,1],"bo")
plt.plot(randomPlot[:,0],randomPlot[:,1],"g^")
plt.show()
km=neighbors.KNeighborsClassifier(n_neighbors=3,p=2)
km.fit(datatrainning,label)
guess=km.predict(randomPlot)
print(guess)