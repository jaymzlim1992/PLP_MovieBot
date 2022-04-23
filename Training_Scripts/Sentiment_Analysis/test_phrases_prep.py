# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:56:28 2022

@author: rggop
"""

import pandas as pd
from sklearn.model_selection import train_test_split

input_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\movie20_processed_cleaned.csv"

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
output_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\phrase_polarity_test_data.csv"

result.to_csv(output_path)