# Import necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

#loading train and test dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.shape

test.shape

train.head()

test.head()

X = train.iloc[:, 1:]
y0 = train.iloc[:, 0]

X.head()


binencoder = LabelBinarizer()
y = binencoder.fit_transform(y0)


X_images = X.values.reshape(-1,28,28)
test_img = test.values.reshape(-1,28,28)

X_images.shape
test_img.shape


plt.imshow(X_images[5])
plt.show()


# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size = 0.2, random_state=90)


test_img2 = test_img/255

test = test_img2.reshape(-1,28,28,1).astype('float32')

model = Sequential()
model.add(Conv2D(32,(4,4),input_shape = (28,28,1),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()

#Validation Split = 0.2
#batch_Size = 90 , 91, 92 
result = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=92, verbose=2)

#Result
history_df = pd.DataFrame(result.history)
history_df.loc[:, ['loss', 'val_loss', 'accuracy', 'val_accuracy']].plot()

scores_test = model.evaluate(X_test, y_test, verbose=0)

pred = model.predict(test)
submit = pd.DataFrame(np.argmax(pred, axis=1), 
                      columns=['Label'], 
                      index=pd.read_csv('../input/digit-recognizer/sample_submission.csv')['ImageId'])


submit.index.name = 'ImageId'
submit.to_csv('conv_model_submission.csv')