# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 18:09:36 2022

@author: rggop
"""

# all imports 
import tweepy
import pandas as pd
import re
import contractions
import nltk
from urlextract import URLExtract
import spacy 
import transformers
import torch
import numpy as np
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import time
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import praw
from praw.models import MoreComments
import statistics

#specify GPU
device = torch.device("cuda")
#url extractor
url_extractor = URLExtract()
#loading the english large model
nlp = spacy.load('en_core_web_lg')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#Define aspects and associated aspect catecories
aspect_dict = {}
aspect_dict["overall"] = ['overall','budget','cinema']
aspect_dict["crew"] = ['act','actor','actress','acting','role','acted'
                       'character','villain','performance','performed','casting',
                       'cast','casted','crew','hero','lead','writers','protagonist',
                       'performance']

aspect_dict["direction"] = ['direction','directing','director','filming','filimmaking',
                           'filmmaker','edition','cinematography',
                           'photography','frame','execution']

aspect_dict["plot"] = ['storyline','story','lines','romance','dialog','dialouges','end'
                        'storyteller','ending','starting','begining','written','fantasy',
                        'storytelling','revenge','tone','climax','concept','moments',
                        'betrayal','plot','drama','dramatic scene','writting','twist',
                        'comedy','jokes','message','transcripts','content',
                        'shock','mystery','final sequence','1st half',
                        '2nd half','second half ','premise','fictional','historical',
                        'history']

aspect_dict["screenplay"] = ['scene','scenery','violence','screenplay','scenario','sets','set',
                        'action','stunt','shot','props','fight','visualization',
                        'costume','battle scenes','fighting scenes',
                        'camera','editing','edited','cuts','script','graphics',
                        'cgi','3d','visual effect','sconces',
                        'visual','effects','special effect','animation',
                        'animated','narrative','narrative structure','setting',
                        'themes','pacing']

aspect_dict["music"] = ['lyric','sound','music','audio','musical','track',
                        'title track','sound effect', 'sound track','score','vocals']


#pre processing tweets
def preprocessing(text):
    
    # convert text to lower case
    text = str(text).lower()
    #remove urls
    urls = url_extractor.find_urls(text)
    for url in urls:
        text = text.replace(url,'')
    #remove emails
    text = re.sub(r'\S+@\S*\s?',' ',text)
    #text = re.sub(r'[A-Za-z0-9]*@[A-Za-z]*\.?[A-Za-z0-9]*',' ',text)
    #change all the mentions to "crew member". Since usually people mentioned in
    #movie review tweets are part of the film either an actor or director etc.
    text = re.sub(r'@\S+','crew',text)
    #expand contractions
    text = contractions.fix(text)
    #remove hashtag
    text = re.sub(r'#\S+',' ', text)
    #remove emojis
    #text = re.sub(r'[^\x00-\x7F]+',' ',text)
    #remove all punct
    text = re.sub('[^A-z0-9]', ' ', text)
    #remove extra white spaces
    text = re.sub(' +', ' ', text)
    return text

# Function to extract Nouns in a Text
def get_noun(lines):
    tokenized = nltk.word_tokenize(lines)
    #extract token whoes POS ends with 'NN' i.e. extracts any type of noun 
    #whether its common or proper
    #additional check to get the aspect 'acting' and 'ending' since sometimes it can be a Verb
    nouns = set([word for (word,pos) in nltk.pos_tag(tokenized) if pos[:2] == 'NN' or word in ['ending','acting']])
    return nouns
#Function to find all aspects and there aspect categories
def get_aspect_categories(nouns,aspect_categories,emb_aspect,noun_str,emb_noun):  
    #initilalize return values 
    aspect_class = []
    noun_class = []
    aspect_name_class = []
    for noun in nouns:
        scores = []
        #Get embeddings of words for noun
        try:
            index = noun_str.index(noun)
            noun_emb = emb_noun[index]
        except:
            noun_emb = nlp(noun)
            noun_str.append(noun)
            emb_noun.append(noun_emb)
        for aspect_emb in emb_aspect:
            #get similarity of both embeddings
            
            similarity_score = aspect_emb.similarity(noun_emb)
            #if aspect_emb.text in noun_emb.text:
            #    print("Aspect : ",aspect_emb,"  Noun :",noun_emb,"  Score:",similarity_score)
            scores.append(similarity_score)
        index = scores.index(max(scores))
        aspect_category = aspect_categories[index][1]
        aspect_name = aspect_categories[index][0]
        #check if similarity score is greater than 60%
        
        if max(scores) > 0.62:
            if noun == 'befor':
                print("befor :",aspect_category,"  Score :",max(scores) )
            aspect_name_class.append(aspect_name)
            aspect_class.append(aspect_category)
            noun_class.append(noun)
            
    return aspect_name_class,aspect_class, noun_class



def test_aspect_method(test_data_path):
    
    aspect_score = []
    pred_aspect = []
    gt_aspect = []
    #flatten the list
    aspect_categories = [[key,x] for key,sublist in aspect_dict.items() for x in sublist]
    # calculate embeddings for each aspect categoriesand noun
    emb_aspect = []
    noun_str = []
    emb_noun = []
    for aspect in aspect_categories:
        emb_aspect.append(nlp(aspect[1]))
    #read data 
    raw_data = pd.read_csv(test_data_path,dtype=str)
    data = pd.DataFrame({'review':raw_data.review.unique()})
    #group data by reviews 
    data['aspect'] = [list(set(raw_data['aspect'].loc[raw_data['review'] == x['review']])) 
    for _, x in data.iterrows()]
    
    #run aspect detection for each review
    precision_num = []
    precision_dim = []
    recall_num = []
    recall_dim = []
    
    for index,row in data.iterrows():
        #print(row[0])
        sentence = row['review']
        aspects_gt = row['aspect']
        sentence_preprocessed = preprocessing(sentence)
        #get all nouns
        noun_list = get_noun(sentence_preprocessed)
        #get pair of aspect word and corresponding aspect category
        
                
        aspect_name_list,aspect_category_list, word_list = get_aspect_categories(
                                                                noun_list,
                                                                aspect_categories,
                                                                emb_aspect,
                                                                noun_str,emb_noun)
        ner = nlp(sentence_preprocessed)
        for ents in ner.ents:
            if ents.label_ == "PERSON":
                word_list.append(ents.text)
        pred_aspect.append(word_list)
        gt_aspect.append(aspects_gt)
        if len(word_list) == 0 and len(aspects_gt) > 0:
            for aspect in aspects_gt:
                aspect_score.append(0)
            recall_num.append(0)
            precision_num.append(0)
            recall_dim.append(0)
            precision_dim.append(0)
        else:
            #calculate precision
            count = 0
            for word in word_list:
                for aspect in aspects_gt:
                    if (word in aspect) or (aspect in word):
                        count = count + 1
            precision_num.append(count)
            precision_dim.append(len(word_list))            
            #calculate recall
            count = 0
            for word in word_list:
                for aspect in aspects_gt:
                    if (word in aspect) or (aspect in word):
                        count = count + 1
            recall_num.append(count)
            recall_dim.append(len(aspects_gt))
            #calculate accuracy
            for aspect in aspects_gt:
                flag = 0
                wor = ""
                a = aspect.lower()
                for word in word_list:
                    w = word.lower()
                    if w in a or a in word:
                        wor = word
                        flag = 1
                if flag == 1 :
                    #print("GT :",aspect, "  Pred :", wor, "    Score :", 1)
                    aspect_score.append(1)
                else:
                    print("GT :",aspect, "  Pred :", word_list , "     Score :", 0)
                    aspect_score.append(0)
            
    #Calculate Percentatge Accuracy
    precision = sum(precision_num)/sum(precision_dim)
    recall = sum(recall_num)/sum(recall_dim)
    f1_score = 2 *((precision * recall)/(precision+recall))
    print("Accuracy = ", statistics.mean(aspect_score) * 100)
    print("Precision = ", precision)
    print("Recall = ",recall )
    print("F1 score =", f1_score)
    #save predictions into csv
    df = pd.DataFrame()
    df["gt_aspect"] = gt_aspect
    df["pred_aspect"] = pred_aspect
    df["review"] = data['review']                                                
    file_name = "aspect_detection.csv"
    df.to_csv(file_name,header=True)
    
#specify path to test data 
test_data_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\movie20_processed_cleaned.csv"
test_aspect_method(test_data_path)
   