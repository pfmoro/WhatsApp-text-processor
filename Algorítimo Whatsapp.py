# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 22:38:12 2019

@author: PC
"""

import re
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance

def Vectorizer(TextVector):
    vectorizer = CountVectorizer()
    vectorizer.fit(TextVector)
    Vectorized= vectorizer.transform(TextVector)
    return Vectorized

def OptimalClusterSize(X):
    # k means determine k
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        print(X.shape)
        print(kmeanModel.cluster_centers_.shape)
        distortions.append(sum(np.min(distance.cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def main(argv):
    #reads txt file with whatsapp messages and creates dataframe containing 
    #one line per message
    results = []
    with open('Conversa do WhatsApp com TEEN8.txt', encoding="utf8") as inputfile:
        
        for line in inputfile:
            
            if len(results) == 0:
                LastLine=0
            else:
                LastLine=len(results)-1  
                
            if line[:10].find('/') != -1:
                results.append(line.strip())
            else:
                results[LastLine] = results[LastLine] + line.strip()       
    inputfile.close
    #end of message dataframe creation

            
    #Creates dataframe with questions
    Questions=[]
    for msg in results:
        if msg[-5:].find('?') != -1:
            Questions.append(msg)
    #end of question dataframe creation
    df_questions = pd.DataFrame(Questions)
    df_questions.to_csv('Teen Coach 8 - perguntas.csv', sep='\t', encoding='utf-8')
    #Loads stopword file and removes it from all questions:
    with open('stopwords.txt', encoding="latin-1") as inputfile:
        CleanQuestions =[]        
        for question in Questions:
            for line in inputfile:
                re.sub(line, '' , question)
            CleanQuestions.append(question)

    inputfile.close
    #end of stopwords removal
    
    #Optimal Cluster size, for K-means:
    VectorizedQuestions=Vectorizer(CleanQuestions)    
    OptimalClusterSize(VectorizedQuestions)
   # re.sub(r'[^\w]', ' ', df)

if __name__ == "__main__":
	main(sys.argv[1:])