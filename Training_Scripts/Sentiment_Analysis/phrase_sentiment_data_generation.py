# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 10:25:52 2022

@author: rggop
"""

#import all packages
from transformers import pipeline
from bs4 import BeautifulSoup
import csv

#build question answering pipeline
qa_pipeline = pipeline("question-answering")

#path to XML file
#input_path = "D:\\PLP_Project\\Sentiment Analysis\\Movie-Reviews-Datasets\\moviesLarge.xml"
input_path = "D:\\PLP_Project\\Sentiment Analysis\\Movie-Reviews-Datasets\\movie20.xml"
#read the data inside xml file
with open(input_path, 'r') as f:
    data = f.read()
#Passing the stored data inside the beautifulsoup parser, storing the returned object
Bs_data = BeautifulSoup(data, "xml")

#Finding all instances of tag sentence 
b_sentence = Bs_data.find_all('sentence')
#iterate and get text with aspect , coresponding aspect term and its polarity.
reviews = []
aspect = []
phrases = []
polarity = []
for sentence in b_sentence:
    #get text value
    text = sentence.find('text').text
    #get all aspect terms if exist
    aspect_terms = sentence.find_all('aspectTerm')
    if len(aspect_terms) < 1:
        continue
    else:
        #for each aspect extract polarity and find the corresponding phrase that
        #describes it in the sentence 
        for aspect_term in aspect_terms:
             term = aspect_term.get("term")
             sense = aspect_term.get("polarity")
             #get the corresponding phrase that discribe aspect using hugging face QA model
             context = text
             question = f"how is {term}"
             result = qa_pipeline(question=question, context=context)
             phrase = result['answer']
             
             # append all results to corresponding list
             reviews.append(text)
             aspect.append(term)
             phrases.append(phrase)
             polarity.append(sense)
             print(f"aspect : {term},   polarity : {sense},   phrase : {phrase}")


#write to csv file
output_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\movie20_processed.csv"
rows = zip(aspect, phrases, polarity, reviews)

with open(output_path, "w",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(("aspect", "phrase","polarity","review"))
    for row in rows:
        writer.writerow(row)




