# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:28:02 2022

@author: rggop
"""

import os
# from google.colab import drive

import numpy as np
import pandas as pd
import json
import random
import time
import datetime
# from rake_nltk import Rake
import pandas as pd
from scipy import spatial

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Read the Database
data_path = "Datasets\\Movie_Recomendation\\fullset_BoW.csv"
df = pd.read_csv(data_path)

dict={
    "GDA" : 'Bag_of_words',
    "GD"  : 'BoW_genre_director',
    "GA"  : 'BoW_genre_actors',
    "DA"  : 'BoW_director_actors',
    "G"   : 'BoW_genre',
    "D"   : 'BoW_director',
    "A"   : 'BoW_actors',
}

#Recommender function
def recommender(genre, director, actors):
  combo = ""
  search_term=""
  if genre != None:
    combo += "G"
    search_term += genre.lower() +" "
  if director != None:
    combo += "D"
    search_term += director.lower() + " "
  if actors != None:
    combo += "A"
    search_term += actors.lower()

  #count = CountVectorizer()
  count = TfidfVectorizer()
  count_matrix = count.fit_transform(df[dict[combo]])

  count_matrix_search=count.transform([search_term])
  cosine_sim = cosine_similarity(count_matrix, count_matrix_search)

  def recommend_search(cosine_sim = cosine_sim):
    recommended_movies = []
    score_series = pd.Series(cosine_sim).sort_values(ascending = False)
    top_5_indices = list(score_series.iloc[:5].index)
    
    #return df.iloc[top_5_indices]
    for i in top_5_indices:
      recommended_movies.append(f"Title: {list(df['Title'])[i]} <br /> Genre: {list(df['Genre'])[i]} <br /> Director: {list(df['Director'])[i]} <br /> Actors: {','.join(df['Actors'][i].split(',')[:3])}")

    return recommended_movies

  return recommend_search(cosine_sim.flatten())

#recommender("crime drame","francis ford coppola", "marlon brando")
#recommender("crime drame","francis ford coppola", None)
#recommender("crime drame",None, "marlon brando")
#recommender(None,"francis ford coppola", "marlon brando")
#recommender("crime drame",None, None)
#recommender(None,"francis ford coppola", None)
#recommender(None,None, "marlon brando")