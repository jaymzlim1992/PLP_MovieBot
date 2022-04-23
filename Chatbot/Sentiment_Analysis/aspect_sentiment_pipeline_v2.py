# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 09:57:50 2022

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
from pathlib import Path
from matplotlib import rcParams
from difflib import SequenceMatcher
rcParams.update({'figure.autolayout': True})
#Flags and other constants
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
aspect_dict["crew"] = ['act','actor','actress','acting','role','portray','acted',
                       'character','villain','performance','performed','casting',
                       'cast','casted','crew','hero','lead','writers','protagonist',
                       'performance']

aspect_dict["direction"] = ['direction','directing','director','filming','filmmaking',
                           'filmmaker','cinematic','edition','cinematography',
                           'photography','frame','making','made','execution']

aspect_dict["plot"] = ['storyline','story','tale','romance','moral','dialog','dialogues','end',
                       'storyteller','ending','starting','beginning','written','fantasy',
                       'storytelling','revenge','tone','climax','concept','moments',
                       'betrayal','plot','drama','dramatic','writting','twist',
                       'comedy','jokes','conclusion','message','transcripts','content',
                       'shock','mystery','final sequence','first half','1st half',
                       '2nd half','second half ','premise','fictional','historical',
                       'history','first act','second act','third act']


aspect_dict["screenplay"] = ['scene','scenery','violence','screenplay','scenario',
                             'sets','set','action','stunt','shot','props','fight',
                             'visualization','costume','battle scenes','fighting scenes',
                             'camera','editing','edited','cuts','script','graphics',
                             'cgi','3d','visual effect','sconces',
                             'visual','effects','special effect','animation',
                             'animated','narrative','setting','themes','pacing',
                             'final sequence']


aspect_dict["music"] = ['lyric','sound','music','audio','musical','track',
                        'title track','sound effect', 'sound track','score','vocals']


aspect_dict["overall"] = ['overall','budget','cinema']

#API KEYS
API_KEY = ""
API_KEY_SECRET = ""
BEARER_TOKEN = ""
ACCESS_TOKEN = ""
ACCESS_TOKEN_SECRET = ""
#Function to fetch reviews from reddit
def get_reddit_reviews(hashtag):
    credentials = {'user_agent': 'MovieBot2022',
                   'client_id' : 'hUQ-A2Si5jNeiUEsEU-DGQ',
                   'client_secret': 'N1SNWWxBkFTuooKBgCRkX6FLPqln2Q',
                   'password': 'GopJMTad@123'
                   }
    #setup reddit client
    reddit = praw.Reddit(user_agent =credentials['user_agent'],
                         client_id =credentials['client_id'],
                         client_secret = credentials['client_secret'],
                         password =credentials['password'],
                         check_for_async=False)
    #select sub reddit
    text = []
    keyword = "Official Discussion - " + hashtag
    sub_reddit = reddit.subreddit('movies')
    #Get the Discussion thread
    reviews = [*sub_reddit.search(keyword, limit=1, time_filter="all")]
    
    movie_fetched = reviews[0].title
    movie_fetched = movie_fetched.replace('Official Discussion -','')
    movie_fetched = movie_fetched.replace('[SPOILERS]','')
    similarity = SequenceMatcher(None, movie_fetched, hashtag.lower()).ratio()
    if similarity <0.5:
        keyword = "Official Discussion: " + hashtag
        reviews = [*sub_reddit.search(keyword, limit=1, time_filter="all")]
        movie_fetched = reviews[0].title
        movie_fetched = movie_fetched.replace('Official Discussion:','')
        movie_fetched = movie_fetched.replace('Official Discussion','')
        movie_fetched = movie_fetched.replace('[SPOILERS]','')
        similarity = SequenceMatcher(None, movie_fetched.lower(), hashtag.lower()).ratio()
        #check proper movie page is fetched
        if similarity < 0.5:
            return pd.DataFrame()
    for review in reviews:
        review.comments.replace_more(limit=0)
        for comment in review.comments.list():  
            comment_str = comment.body
            #split comment into sentences 
            comment_sent = nltk.tokenize.sent_tokenize(comment_str)
            #appent to final list
            text.append(comment_sent)
    
        
    #flatten final list
    text = [x for sublist in text for x in sublist]
    #creating a dataframe  
    reddit_df = pd.DataFrame({'text' : text})
    
    
    return reddit_df
#Function to fetch tweets from twitter
def get_tweets(hashtag,api_key,api_key_secret,access_token,access_token_secret,
               tweet_count):
    
    #authenticate using credentials
    auth = tweepy.OAuthHandler(api_key,api_key_secret)
    #flag to remove retweets
    flag = " -filter:retweets"
    hashtag = hashtag + flag
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit= True)
    
    #fetching tweets holding specific hastag
    tweet_text,tweet_date = [],[]
    for tweet in tweepy.Cursor(api.search_tweets,q = hashtag,lang = "en",count =tweet_count,
                               tweet_mode='extended').items(tweet_count):
        tweet_text.append(tweet.full_text)
        tweet_date.append(tweet.created_at)
    #creating a dataframe  
    tweets_df = pd.DataFrame({'text' : tweet_text,'date' : tweet_date})
    
    
    return tweets_df


#pre processing tweets
def preprocessing(twitter_content):
    pre_processed = []
    # convert text to lower case
    for text in twitter_content['text']:
        text = str(text).lower()
        #remove urls
        urls = url_extractor.find_urls(text)
        for url in urls:
            text = text.replace(url,'')
        #remove emails
        text = re.sub(r'\S+@\S*\s?',' ',text)
        #text = re.sub(r'[A-Za-z0-9]*@[A-Za-z]*\.?[A-Za-z0-9]*',' ',text)
        #remove twitter mentions
        text = re.sub(r'@\S+','',text)
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
        pre_processed.append(text)
    twitter_content['pre_processed'] = pre_processed
    
    return twitter_content

#load sentiment analysis models and QA model
def init_models(model_path):
    sent_model = torch.load(model_path)
    sent_model.eval()
    sent_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base",normalization=True)
    #build question answering pipeline
    qa_pipeline = pipeline("question-answering",device=0)
    return sent_model, sent_tokenizer,qa_pipeline

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
            scores.append(similarity_score)
        index = scores.index(max(scores))
        aspect_category = aspect_categories[index][1]
        aspect_name = aspect_categories[index][0]
        #check if similarity score is greater than 60%
        if max(scores) > 0.6:
            aspect_name_class.append(aspect_name)
            aspect_class.append(aspect_category)
            noun_class.append(noun)
    return aspect_name_class,aspect_class, noun_class
#function to identify aspect category and phrases describing it
def get_aspect_category_phrase(preprocess_content,aspect_dict,QA_model): 
    #flatten the list
    aspect_categories = [[key,x] for key,sublist in aspect_dict.items() for x in sublist]
    # calculate embeddings for each aspect categoriesand noun
    emb_aspect = []
    noun_str = []
    emb_noun = []
    for aspect in aspect_categories:
        emb_aspect.append(nlp(aspect[1]))
    #initialize empty lists 
    aspect_name = []
    aspect_category = []
    aspect_word = []
    phrase = []
    
    #initialize list to store QA model variables
    qa_aspect = []
    qa_aspect_category = []
    qa_aspect_word =[]
    qa_phrase = []
    question = []
    context = []
    
 
    for text in preprocess_content['pre_processed']:
        #get all nouns
        noun_list = get_noun(text)
        #get pair of aspect word and corresponding aspect category
        aspect_name_list,aspect_category_list, word_list = get_aspect_categories(
                                                                noun_list,
                                                                aspect_categories,
                                                                emb_aspect,
                                                                noun_str,emb_noun)
        #run NER on each sentence to identify names of any crew members (actors,writters)
        #mentioned and add them to the aspect list        
        ner = nlp(text)
        for ents in ner.ents:
            if ents.label_ == "PERSON":
                aspect_name_list.append('crew')
                aspect_category_list.append('cast')
                word_list.append(ents.text)
        #text without any aspect categories is considered as overall category.
        # and the entire text is used to estimate polarity of this category
        if len(aspect_category_list) == 0:
            aspect_name.append("overall")
            aspect_category.append("overall")
            aspect_word.append("overall")
            phrase.append(text)
        else:
            #create batch of input to QA model
            
            for aspect,aspect_c, word in zip(aspect_name_list,aspect_category_list, word_list):
                #format question
                 q = f"how is {word}"
                 qa_aspect.append(aspect)
                 qa_aspect_category.append(aspect_c)
                 qa_aspect_word.append(word)
                 question.append(q)
                 context.append(text)

           
    #find the phrases for each aspect category

    qa_result = QA_model(question = question, context = context,batch_size = 128)
    qa_phrase = [qa['answer'] for qa in qa_result]


    #creating a dataframe  
    aspect_phrase_df = pd.DataFrame({'aspect' : aspect_name,
                                     'aspect_category' : aspect_category,
                                     'aspect_word' : aspect_word,
                                     'phrase' : phrase})
    qa_aspect_phrase_df = pd.DataFrame({'aspect' : qa_aspect,
                                        'aspect_category' : qa_aspect_category,
                                        'aspect_word' : qa_aspect_word,
                                        'phrase' : qa_phrase})
    #merge two data frames 
    aspect_phrase_df = aspect_phrase_df.append(qa_aspect_phrase_df)
    
    
    
    return aspect_phrase_df
                 
        
#function to perform sentiment analysis
def perform_sentiment_analysis(sent_model,sent_tokenizer,sentiment_content):
    #perform preprocessing
    #Tokenize and encode sequence in the test set
    MAX_LENGTH = 25
    polarity = []
    # get list of phrases
    #for phrase in sentiment_content['phrase']:
    phrase = sentiment_content['phrase']
    tokens = sent_tokenizer.batch_encode_plus(phrase, max_length = MAX_LENGTH,
                                  padding=True, truncation = True,
                                  return_token_type_ids= False)

    #convert integer tokens to pytorch tensors
    text_seq = torch.tensor(tokens['input_ids'])
    text_mask = torch.tensor(tokens['attention_mask'])
    
    
    #Define Data Loaders for train set
    #define batch size
    batch_size = 128
    #combine tensors 
    train_dataset = TensorDataset(text_seq,text_mask)
    #data loader for train set
    train_dataloader = DataLoader(train_dataset,batch_size= batch_size)
    
    #get predictions for  data
    with torch.no_grad():
        for seq, mask in train_dataloader:
            result = sent_model(seq.to(device),mask.to(device))[0]
        
            result = result.detach().cpu().numpy()
            result = np.argmax(result,axis = 1)
            polarity.append(result)
    #flatten polarity list 
    polarity = [ x for sublist in polarity for x in sublist]
    sentiment_content['polarity'] = polarity
    
    return sentiment_content      
        
#Function to visualize result
def visualize_sentiment(aspect_score):
    
    result = [[k ,v['positive']*100/(v['positive'] + v['negative']),
               v['negative']*100/(v['positive'] + v['negative'])] if (v['positive'] + v['negative']) > 0 else [k,0,0] 
              for k,v in aspect_score.items()]
    
    #plot the bar plot across all aspects
    
    aspects_df = pd.DataFrame(result, columns= ['aspect', 'positive', 'negative'])
    # plot a Stacked Bar Chart using matplotlib
    aspects_df.plot(
        x = 'aspect',
        kind = 'barh', 
        stacked = True, 
        title = 'Sentiment Analysis Result',
        color={"negative": "red", "positive": "green"},
        mark_right = True)
  
    df_total = aspects_df["positive"] + aspects_df["negative"]
    df_rel = aspects_df[aspects_df.columns[1:]].div(df_total, 0)*100
  
    for n in df_rel:
        for i, (cs, ab, pc) in enumerate(zip(aspects_df.iloc[:, 1:].cumsum(1)[n], 
                                         aspects_df[n], df_rel[n])):
            if pc > 0:
                plt.text(cs - ab / 2, i, str(np.round(pc, 1)) + '%', 
                 va = 'center', ha = 'center')
                
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.savefig('static\\sentiment_graph.jpg')
#Function to do Twitter sentiment analysis of movie
def aspect_sentiment_analysis(hashtag,sent_model,sent_tokenizer,QA_model):
    #extracts tweets regarding the hashtag from twitter
    #review_content = get_tweets(hashtag, API_KEY, API_KEY_SECRET, ACCESS_TOKEN,
    #                          ACCESS_TOKEN_SECRET, tweet_count=10000)

    print("TITLE : ",hashtag)    
    #extract reviews from reddit
    review_content = get_reddit_reviews(hashtag)
    
    #Failed to get information about the movie returning failure status
    if review_content.empty:
        return 2 
    #preprocess the twitter content
    preprocess_content = preprocessing(review_content)
    #find aspect aspect category noun and phrase
    sentiment_content = get_aspect_category_phrase(preprocess_content,aspect_dict,QA_model)
    #find polarity of each aspect based phrases 
    sentiment_content = perform_sentiment_analysis(sent_model,sent_tokenizer,sentiment_content)
    aspects = aspect_dict.keys()
    aspect_score = {asp : {'positive':0 , 'neutral':0, 'negative':0} for asp in aspects}
    #compute final scores
    for aspect_n, polarity in zip(sentiment_content['aspect'],sentiment_content['polarity']):
        if polarity == 0:    
            aspect_score[aspect_n]['negative'] +=1
        elif polarity == 1:
            aspect_score[aspect_n]['neutral'] +=1
        else:
            aspect_score[aspect_n]['positive'] +=1
    visualize_sentiment(aspect_score)
    #Finished Aspect based sentiment analysis return Sucess status
    return 1


    
    

