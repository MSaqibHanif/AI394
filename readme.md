# AI-394
### PROJECT MEMBERS ###
StdID | Name
------------ | -------------
**63770** | **Muhammad Saqib** 
60008 | Humayun Iqbal

# Abstract #
We implemented the following technologies:

## 1. Cross Validation ##
**Cross Validation** is a resampling procedure in which the dataset is randomly split up into 'K' groups. K refers to the number of groups that a given data sample is to be split into. For splitting data we have to give a range of specific value. This procedure is often called K-fold cross validation. We used this procedure to split the dataset into 5x3.

## 2. Naive Bayes ##
**Naive Bayes** is a supervised machine learning algorithm that’s typically used for classification problems. We used it for solving multi-class classification problems.

## 3. SVM ##
**Support Vector Machine or SVM** is a supervised machine learning algorithm which manipulates the kernel into data transformation. We use this algorithm to train our model and predict possible outcomes.

## 4. Training Data ##
**For Training Data** we used train.csv file. For reading our training data. Pandas is mostly used to help with data organization and visualization. Since you can put data into dataframes and datasets. We used pandas library tool here to read data from train.csv file.

## 5. Linear Regression ##
**Linear Regression** is a mathematical model which is used to predict the value of a variable based on the value of another variable. We used this model for better prediction of our training results.

## 6. Convolution ##
**Convolution** is the most common type of artificial neutral network which is used in image recognition and processing to process pixel data. Convolution is used in the 2D convolution layer. It is often abbreviated as "conv2D". It has a small kernel, essentially a window of pixel values, that slides along those two dimensions. The RGB channel is not handled as a small window of depth, but rather, is obtained from beginning to end, first channel to last. That is, even a convolution with a small spatial window of 1x1, which takes a single pixel spatially in the width/height dimensions, would still take all 3 RGB channels. We used it to produce clearer images on it to recognize the digits clearly.

## 7. KNN ##
**KNN** algorithm is one of the simplest classification algorithms. KNN algorithm is a frequently used machine learning algorithm. KNN is a non-parametric, lazy learning algorithm. Its purpose is to use a database in which the data points are separated into several classes to predict the classification of a new sample point.The KNN algorithm can compete with the most accurate models because it makes highly accurate predictions. We used it to acquire the highest accuracy in test results.

## How we tweaked the parameters ## 
```py
Multinb = MultinomialNB()

param_grid = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0],
              'fit_prior': [True,False]}

mnb_GS = GridSearchCV(Multinb, param_grid, cv=5, verbose=2, n_jobs=1)

tiny_model.fit(X_train, y_train, epochs=5, batch_size=32)

test_loss, test_acc = tiny_model.evaluate(X_test,  y_test, verbose=2)
```
For the first algorithm, We used the **Naive-Bayes classifier**. Assuming that all predictors have an equal effect on the outcome and one predictor variable does not affect the presence of another, the algorithm seems like a good option.
Since our case is about a classification problem with multple possible outcomes (digits 0 to 9), the **Multinomial** type of **Naive-Bayes Classifier** would be a better fit compared to the other 2 types. The Bernoulli type recommends a binary outcome whereas the Gaussian type requires predictors to be continous. 
From sklearn, we also import our different libraries to help us in training model and train that dataset efficently.
```py
train_images = np.reshape(X_train, (len(X_train),28,28))
test_images = np.reshape(X_test, (len(X_test),28,28))
```
At this point, we tried to filter out and reshape the images in 5x5 which give us the significantly better results and more accuracy.
```py
def layer_size(X, Y):
    
    n_x = X.shape[1]
    n_h = 4
    n_y = Y.shape[1]
    
    return (n_x, n_h, n_y)
```
   
Here, we started using the neural network to train our dataset. We initialized the function layer to return the size and tweaked it to fit our model. 
 
```py
train,test = train_test_split(train,test_size = 0.25)
```
At this parameter, we simply split our data to train it seperately by using **Cross-Validation** to split data.
## Few words about Techniques ##
We have implemented 5 techniques to improve the training of our model and produce better results. **(SVM, KNN, Neural, Linear Regression, Convolution)**. By using these 5 techniques, we understood how exactly we can improve training our model to produce more accurate results.
Firstly, we used SVM which gave us maximum accuracy of 36.7% on first attempt. Then, we applied linear regression which gave us about 49.7% accuracy. To improve it further, we applied some neural network functions which gave us the highest accuracy of 63%. The accuracy was significantly better than our past results. Finally, we applied KNN technique on our dataset, and after several tries the test accuracy reached 99.26% after which we stopped training.
So, after applying all of the above techniques we can conclude saying that KNN is one of the best techniques for training model and predicting dataset with the highest accuracy.

## Kaggle Submission ##

Although implementation of KNN technique gave us 99% test accuracy. To achieve the perfect score for Kaggle competition, we would've opted for implementation of Neural Network. To view more details about our submission, Click on [Kaggle](https://www.kaggle.com/muhammadsaqibshah/notebook3508461fa4).


<img src="/final project/FinalScore.PNG" alt="Highest Kaggle score"/>

This repository contains assignments and project submitted to AI course offered in Spring 2021 at KIET.
