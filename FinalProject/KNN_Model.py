
# Import necessary modules
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from skimage import exposure
import cv2

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.data.shape)
print(digits.images.shape)

# Display the 1011th image using plt.imshow().
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
test_size=0.1, random_state=84)

print("training data points: {}".format(len(X_train)))
print("validation data points: {}".format(len(y_test)))
print("testing data points: {}".format(len(y_train)))

kVals = range(1, 30, 2)
accuracies = []

for k in range(1, 30, 2):
          # train the k-Nearest Neighbor classifier with the current value of `k`
          model = KNeighborsClassifier(n_neighbors=k)
          model.fit(X_train, y_train)
          # evaluate the model and update the accuracies list
          score = model.score(X_test, y_test)
          print("k=%d, accuracy=%.2f%%" % (k, score * 100))
          accuracies.append(score)

i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
accuracies[i] * 100))

model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("EVALUATION ON TESTING DATA")
print(classification_report(y_test, predictions))

print ("Confusion matrix")
print(confusion_matrix(y_test,predictions))

for i in np.random.randint(0, high=len(y_test), size=(5,)):
         # grab the image and classify it
         image = y_test[i]
         prediction = model.predict([image])[0]
         
         # show the prediction
    
imgdata = np.array(image, dtype='float')
pixels = imgdata.reshape((8,8))
pltT.imshow(pixels,cmap='gray')
pltT.annotate(prediction,(3,3),bbox={'facecolor':'white'},fontsize=16)
print("i think the digit is : {}".format(prediction))
#cv2.imshow("image", image)
pltT.show()
cv2.waitKey(0)