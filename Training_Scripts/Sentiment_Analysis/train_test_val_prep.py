# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 15:21:04 2022

@author: rggop
"""

import pandas as pd
from sklearn.model_selection import train_test_split

input_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\moviesLarge_processed_cleaned.csv"

data = pd.read_csv(input_path,dtype=str)

#select only phrase and polarity columns for processing
result = pd.concat([data['phrase'],data['polarity']], axis=1, join='inner')

#convert all string values in data frame to lower case
result['phrase'] = result['phrase'].str.lower()

print("Total data before removing duplicates : ",result.shape[0])


#eleminate duplicate combination of phrase and polarity
result = result.drop_duplicates()
print("Total data after removing duplicates : ",result.shape[0])

# write the unique phrases and polarity to a file
output_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\unique_phrases_polarity.csv"

result.to_csv(output_path)

#read values from the kaggle move sentence data
#append it to phrase and polarity dataframe

text_file_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\imdb_labelled.txt"
for line in open(text_file_path,'r'):
    split = line.strip().split('\t', 1)
    if split[1] == '0':
        new_row = {'phrase': split[0], 'polarity' : 'negative'}
    else:
        new_row = {'phrase': split[0], 'polarity' : 'positive'}
    result = result.append(new_row, ignore_index=True)
    
    
print("Total data after adding kaggle data : ",result.shape[0])
    
#write the final data to a file
#result = result.sort_values(by=['phrase'], ascending=True)
output_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\final_phrase_polarity_data.csv"

result.to_csv(output_path)

#split the data into train, val
train, val = train_test_split(result, test_size=0.1)

print("train data length : ",train.shape[0])
print("val data length : ",val.shape[0])


#write train  and val data into a file
output_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\train_data.csv"
train.to_csv(output_path)

output_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\val_data.csv"
val.to_csv(output_path)
