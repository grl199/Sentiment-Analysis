#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:11:42 2021

@author: guille
"""


import time
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#df=pd.read_csv('tidy_test.csv', sep='|')


'''
Outputs a csv with clean phrases
'''
def cleaning(n,file):
    start = time.time()
    data=pd.read_csv(file, sep='|')
    corpus=[]
    for i in range(n):
        review= data['phrase'][i]
        #review = re.sub('[^a-zA-Z!]', ' ', data['phrase'][i]) #Regular expressions
        review = review.lower() #Lower case
        review = review.split() #split
        ps = PorterStemmer()
        review=[ps.stem(word) for word in review if not word in stopwords.words('english')] #Remove stopwords
        review=' '.join(review) #Join the words with blank spaces
        corpus.append(review)
    
   
    output=pd.DataFrame({'phrase':corpus , 'sentiment values' : data['sentiment values'].values[:n]})
    end = time.time()
    print('Elapsed time:  ' + str("{:.2f}".format(end - start))+ ' seconds,   '+  str("{:.2f}".format((end - start)/60))+ ' minutes,  ' + str("{:.2f}".format((end - start)/3600))+  ' hours')
    output.to_csv(file[:-4]+'_clean.csv')
    
    
def cleaning_bis(file):
    #Takes a pandas column of phrases and cleans it :)
    df=pd.read_csv(file,sep='|')
    phrases=df.phrase
    phrases = phrases.apply(lambda x: x.replace(" 've", ""))
    phrases = phrases.apply(lambda x: x.replace(" 's", "s"))
    phrases = phrases.apply(lambda x: x.replace(" 'll", "ll"))
    phrases = phrases.apply(lambda x: x.replace(" 're", "re"))
    phrases = phrases.apply(lambda x: x.replace(" 'd", "d"))
    phrases = phrases.apply(lambda x: x.replace(" n't", "nt"))
    phrases = phrases.apply(lambda x: x.replace("-", " "))
    phrases = phrases.apply(lambda x: re.sub(r'[^A-Za-z0-9 ]+', '', x))
    phrases = phrases.apply(lambda x: x.replace("  ", " "))
    phrases = phrases.apply(lambda x: x.strip())
    phrases = phrases.apply(lambda x: x.lower())
    
    output=pd.DataFrame({'phrase':phrases , 'sentiment values' : df['sentiment values'].values})
    output.to_csv(file[:-4]+'_clean.csv')

    
cleaning(10000,"tidy_train.csv")
cleaning_bis("tidy_train.csv")

df=pd.read_csv('tidy_train_clean.csv')


    