import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ShuffleSplit

from tensorflow import keras

#read dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#describing the data 
train.describe()

train.info()

#displaying top 10 Data points
train.head(10)

#cross validation for splitting
train,test=train_test_split(train,test_size=0.25)
#displaying the head of test dataset
test.head()

rn = range(1,26) 

kf3 = KFold(n_splits=4, shuffle=False) #hipper parameters


for train_index, test_index in kf3.split(rn):
    print(train_index, test_index)
	
test_x = test.values

train = train / 255.
test  = test  / 255.

#training examples here we have 
shape_x = train_x.shape
shape_y = train_y.shape

m = train_y.shape[0]

print ('Shape of train X : ' + str(shape_x))
print ('Shape of train Y : ' + str(shape_y))
print ('I have m = %d training examples!' % (m))


def layer_size(X, Y):
    
    n_x = X.shape[1]
    n_h = 4
    n_y = Y.shape[1]
    
    return (n_x, n_h, n_y)

def initialise_parameter(n_x, n_h, n_y):
    
    np.random.seed(0)
    
    W1 = np.random.randn(n_h[0], n_x) * 0.1
    b1 = np.zeros(shape=(n_h[0], 1))
    
    W2 = np.random.randn(n_h[1], n_h[0]) * 0.1
    b2 = np.zeros(shape=(n_h[1], 1))
    
    W3 = np.random.randn(n_y, n_h[1]) * 0.1
    b3 = np.zeros(shape=(n_y, 1))
    
    assert(W1.shape == (n_h[0], n_x))
    assert(b1.shape == (n_h[0], 1))

    assert(W2.shape == (n_h[1], n_h[0]))
    assert(b2.shape == (n_h[1], 1))
    
    assert(W3.shape == (n_y, n_h[1]))
    assert(b3.shape == (n_y, 1))
    
    parameters = {"W1": W1, 
                  "b1": b1, 
                  "W2": W2, 
                  "b2": b2, 
                  "W3": W3, 
                  "b3": b3
                 }
    
    return parameters

tiny_model = keras.Sequential([   keras.layers.Dense(16, activation='relu', input_shape=(784,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
                          ])

#optimizer
tiny_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])


range_class = np.arange(10)
y = np.asfarray(train.iloc[:,0])

train_x = train.iloc[:,1:].values
train_y = np.array([(range_class==label).astype(np.float) for label in y])

test_x = test.values

train = train / 255.
test  = test / 255.


X = np.array(train.drop('label', axis=1))
y = np.array(train['label'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)


print('Each pixel has a value between 0 and ' + str(X_train.max()))



# two-dimensional arrays (28x28 pixels) 
train_images = np.reshape(X_train, (len(X_train),28,28))
test_images = np.reshape(X_test, (len(X_test),28,28))

# Plot the first 45 train images
plt.figure(figsize=(15,15))
for i in range(45):
  plt.subplot(5, 5, i+1)
  plt.imshow(train_images[i], cmap='binary')
  plt.text(0,28, y_train[i], color='green')
  plt.xticks([])
  plt.yticks([])
plt.show()


plt.figure()
plt.imshow(train_images[0])
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.show()


from sklearn.naive_bayes import MultinomialNB

X_train.shape
y_train.shape
X_test.shape
Multinb = MultinomialNB()

param_grid = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0],
              'fit_prior': [True,False]}

mnb_GS = GridSearchCV(Multinb, param_grid, cv=5, verbose=2, n_jobs=1)

tiny_model.fit(X_train, y_train, epochs=5, batch_size=32)

test_loss, test_acc = tiny_model.evaluate(X_test,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)
