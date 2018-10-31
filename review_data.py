# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 18:23:07 2018

@author: aayush
"""



#here i have used xgb & sklearn api rather than writing the whole algorithm myself


#importing Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math


#file path, i copied the review data from potential datasets spreadsheet and stored it in a csv file

file_path = os.path.abspath('C:/Aayush/Innovaccer/review_data.csv')

#loading the dataset

train = pd.read_csv(file_path)

train.head()

# examining the dataset


print('Data types of features of the dataset')
print(train.dtypes)
print()
print("Null values in the dataset")
print(train.isnull().sum())
print("an overview of the dataset")
print()
print(train.head(4))
print()
print("Description of the dataset")
print(train.describe())



# it is found that the column 'Member years' has a negative value which is absurd 
# so assuming it as a missing value and removing the row containing negative value of member years

for i in range(len(train['Member years'])):
    if train['Member years'][i]<0:
        train = train.drop([i], axis = 0)
        

# separating numerical data and categorical data from the train
num_data = train.select_dtypes(include =  [np.number])

cat_data = train.select_dtypes(exclude = [np.number])


# checking correlation of numerical data

correlation = num_data.corr()
sns.heatmap(correlation)
print()
print("Correlation with respect to the target variable :")
print()
print(correlation['Score'])


# performing anova'analysis of variance' to check for important categorical features
# checking for disparity scores and plotting them


categorical_columns = [f for f in train.columns if train.dtypes[f] == 'object']
def anova(frame):
    anv = pd.DataFrame()
    anv['features'] = categorical_columns
    pvals = []
    for c in categorical_columns:
           samples = []
           for cls in frame[c].unique():
                  s = frame[frame[c] == cls]['Score'].values
                  samples.append(s)
           pval = stats.f_oneway(*samples)[1]
           pvals.append(pval)
    anv['pval'] = pvals
    return anv.sort_values('pval')

cat_data['Score'] = train.Score.values
k = anova(cat_data) 
k['disparity'] = np.log(1./k['pval'].values) 
sns.barplot(data=k, x = 'features', y='disparity') 
plt.xticks(rotation=90) 
plt


# we see that in numerical data there is Nr.rooms which has the least value but its modulus is greatest hence 
# it has the highest negative correlation
# and in categorical data hotel stars and hotel name has the highest values of disparity thereby indicating the highest importance
# in calculating the output

# now creating some new features from given data 'Feature Engineering'

# first lets encode categorical values in numbers


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cat_data = cat_data.apply(le.fit_transform)


# first lets create an amenity scale of the hotel on the basis of given amenities such as swimming pool, yoga classes, club etc

train['amenities scale'] = cat_data['Swimming Pool'] + cat_data['Exercise Room'] + cat_data['Yoga Classes'] + cat_data['Free Wifi']


# the review given by a user also depends on a factor that out of the times he has reviewed something, how many times he gave a 
# positive review, so lets create a feature %help = helful votes/nr. of reviews

train['%help'] = num_data['Helpful votes']/ num_data['Nr. reviews']


# traveler type of 'Couples' and 'Friends' have a higher chance of going to a club if a hotel is failitated by one and they might 
# end up giving a good review to such a hotel

train['clubbed'] = ((train['Traveler type'] == 'Friends').astype(np.int64) + (train['Traveler type'] == 'Couples').astype(np.int64))*((train['Club'] == 'YES').astype(np.int64))


# a particular hotel may be suited more to a couple stay than a business travel, thus mapping a new feature based on the traveler 
# type and the hotel's name

train['Hotel by type'] = cat_data['Traveler type']*cat_data['Hotel name']


# there is a good probability that a hotel might serve good asian food but bad italian food, therefore a guy from india 
# will give it a higher score than a guy from europe, also there is a probability that a particular hotel may be famous in some continent

# thus mapping a new feature based on the hotel's name and user's continent

train['Hotel by continent'] = cat_data['User continent'] * cat_data['Hotel name']

# since there has been addition of new columns in the dataset therefore extracting out numerical values again

num_data = train.select_dtypes(include = [np.number])

# again checking for the correlation of our new features

corr1 = num_data.corr()
print('Correlation of features with respect to score:')
print(corr1['Score'])

sns.heatmap(corr1)


# our new features had higher values than the features given, yet correlation values and disparity values dont have such a 
# great value and again data points are less


# first lets create an svm 

from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', C = 1, max_iter = 100000)


# lets preprocess the data accordingly
# since Nr. reviews, Nr. Hotel reviews and Helpful votes have a high value of correlation with each other and a very small value 
#  of correlation with the target variable thus dropping these features

num_data = num_data.drop(['Helpful votes', 'Nr. reviews', 'Nr. hotel reviews'], axis = 1)

#Since svms are sensitive to noisy data and these features had low disparity scores
cat_data = cat_data.drop(['Review weekday', 'Review month', 'Club', 'Yoga Classes', 'Period of stay'], axis = 1)


#taking out the target variable
y = num_data['Score'].iloc[:].values

#dropping out the target variable from the numerical data
num_data = num_data.drop(['Score'], axis = 1)


# now performing one hot encoding on the categorical data, 
# they were previously encoded into numerical values

from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
onehot.fit(cat_data)
cat_data = onehot.transform(cat_data).toarray()


# converting dataframes to numpy arrays
X_num = num_data.iloc[:,:].values
X_cat = cat_data # cat_data was previously converted into a numpy array while one hot encoding


#concatenating these values in a single feature matrix
X = np.concatenate((X_num,X_cat), axis = 1)


#splitting into training and test set

from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size = 50, random_state = 34)


# validating the model using k fold cross validation strategy
from sklearn.model_selection import cross_val_score as cv
accuracy_of_svm = cv(estimator = svm, X = X_train, y = y_train)

#printing out the mean accuracy
print("acuracy of svm is = ",accuracy_of_svm.mean())


# predictions of svm on the test set

svm.fit(X_train, y_train)

#making predictions of svm on test set
y_hat = svm.predict(X_test)

# calculating accuracy
accuracy = (y_hat == y_test).astype(np.int).sum()/len(y_hat)

print('accuracy of svm = ', accuracy)


# thats not a good score but we have scarcity of data and skewness of classes and non correaltion of features
# lets try to reduce the number of classes


# new mapping of classes is such that if score is 1 or 2 it will gove it class 0 , if 3 class 1 is given, 
# if 4 or 5 class 2 is given


for i in range(len(train.Score)):
    if i == 75:
        a = 2 #useless command just to escape from keyerror
    
    else:
        if train.Score[i] == 1 or train.Score[i] ==2 :
            train.Score[i] = 0 # poor review
        elif train.Score[i] == 3 or train.Score[i] == 4:
            train.Score[i] = 1 # average review
        elif train.Score[i] == 5:
            train.Score[i] = 2 # good review
    
        
# this time we will create an ensembled model of xgb classifier

#first setting out the new target variable
y = train.Score.iloc[:].values


#separating out the categorical and numerical features from data
num_data = train.select_dtypes(include = [np.number])
cat_data = train.select_dtypes(exclude = [np.number])



#one hot encoding the categorical features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le2 = LabelEncoder()
cat_data = cat_data.apply(le2.fit_transform)
onehot2 = OneHotEncoder()
onehot2.fit(cat_data)
cat_data = onehot2.transform(cat_data).toarray()


num_data = num_data.drop(['Score'], axis = 1)


#converting dataframes to numpy arrays, cat_data was already converted to a numpy array while being one hot encoded
xnum = num_data.iloc[:,:].values
xcat = cat_data


#merging into one feature matrix
x = np.concatenate((xnum, xcat), axis = 1)



# splitting into training and test set
from sklearn.model_selection import train_test_split as tts
x_train, x_test, y_train, y_test = tts(x , y, test_size = 40, random_state = 32)


#importing xgboost classifier

from xgboost import XGBClassifier

model1 = XGBClassifier(random_state = 36)
model2 = XGBClassifier(random_state = 89)
model3 = XGBClassifier(random_state = 56)
model4 = XGBClassifier(random_state = 32)
model5 = XGBClassifier(random_state = 11)


# importing ensembler
from sklearn.ensemble import VotingClassifier
xgb = VotingClassifier([('xgb1', model1), ('xgb2', model2), ('xgb3', model3), ('xgb4', model4), ('xgb5', model5)], voting = 'hard')

#validation of model using k fold cross validation 
from sklearn.model_selection import cross_val_score as cv
accuracy_of_ensembler = cv(estimator = xgb, X = x_train, y = y_train)
print("accuracy of ensembled xgboost", accuracy_of_ensembler.mean())




#checking the accuracy
print(accuracy_of_ensembler.mean())

# fitting the xgb model on training set
xgb.fit(x_train, y_train)

#making predictions
yhat = xgb.predict(x_test)

#checking accuracy on test set
acc = (yhat == y_test).astype(np.int).sum()/len(yhat)


print("accuracy on the test set = ", acc)


'''Thus we see that due to lack of data points and irrelevance of features, svm cannot perform well on the dataset, but if we reduce the number of classes as their distribution is not proper, we manage to get a decent classifier,considering the odds, using an xgboost classifier. We didnt do any feature dropping on this model as xgboost algorithm is immune to noisy data. Final model was an ensemble of 5 Xgboost models

The features which were important were shown earlier by using disparity scores and correlations and their plots. Thus we can conclude that our most important features were

Hotel Stars
Hotel Name
Amenities Scale
Hotel by continent
The below two features were engineered or derived from given features'''


cat_data = train.select_dtypes(exclude = [np.number])
num_data = train.select_dtypes(include = [np.number])

#adding scores to cat data
cat_data['Score'] = train.Score.values

#checking for disparity scores with new classes
anv2 = anova(cat_data)
anv2['disparity'] = np.log(1./anv2['pval'].values) 
sns.barplot(data=anv2, x = 'features', y='disparity') 
plt.xticks(rotation=90) 
plt



#code for correlation plot
corr3 = num_data.corr()
sns.heatmap(corr3)
plt
print("Correlation scores", corr3['Score'])
    
        



