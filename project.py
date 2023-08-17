#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:55:33 2021

@author: guille
"""

import numpy as np
import pandas as pd
import seaborn as sns


labels = pd.read_csv('sentiment_labels.txt',sep='|')

labels.loc[labels['sentiment values']<=0.2,'sentiment values']=1
labels.loc[labels['sentiment values']<=0.4,'sentiment values']=2
labels.loc[labels['sentiment values']<=0.6,'sentiment values']=3
labels.loc[labels['sentiment values']<=0.8,'sentiment values']=4
labels.loc[labels['sentiment values']<1,'sentiment values']=5


labels['sentiment values'].describe()
labels['sentiment values']

sns.distplot(labels['sentiment values'])


sentences=pd.read_csv('datasetSentences.txt',sep='	')
dic=pd.read_csv('dictionary.txt',sep='|',header=None)
dic.columns=['No','Index']



split_index=pd.read_csv('datasetSplit.txt',sep=',')
split_index.head


dic=dic.sort_values(by='Index')

df=pd.DataFrame({"Phrase" : dic.No.tolist(), 'Label' : labels['sentiment values'].tolist()})
data=pd.DataFrame({"Phrase" : dic.No.tolist(), 'Label' : labels['sentiment values'].tolist()})



import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# ps = PorterStemmer()
# phrase='This is a clear example of how cleaning works'
# phrase = phrase.lower() #Lower case
# phrase = phrase.split() #split
# ps = PorterStemmer()
# phrase=[ps.stem(word) for word in phrase if not word in stopwords.words('english')] #Remove stopwords
# phrase=' '.join(phrase) #Join the words with blank spaces


corpus=[]
n=10000
for i in range(n):
    review = re.sub('[^a-zA-Z!]', ' ', data['Phrase'][i]) #Regular expressions
    review = review.lower() #Lower case
    review = review.split() #split
    ps = PorterStemmer()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')] #Remove stopwords
    review=' '.join(review) #Join the words with blank spaces
    corpus.append(review)
    

df=data.iloc[:n,:]

#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values


# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Gaussian naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30, criterion = "entropy", random_state = 0, max_leaf_nodes=35)
classifier.fit(X_train, y_train)


#Predicted values on the validation set
y_pred=classifier.predict(X_valid)


from sklearn.metrics import accuracy_score,f1_score,zero_one_loss,confusion_matrix,precision_score,plot_confusion_matrix
accuracy_score(y_valid,y_pred)
precision_score(y_valid,y_pred,average='weighted')
f1_score(y_valid,y_pred)
zero_one_loss(y_valid,y_pred)
confusion_matrix(y_valid,y_pred)

plot_confusion_matrix(logreg,X_valid,y_valid)







