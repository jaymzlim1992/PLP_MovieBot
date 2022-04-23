# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:59:39 2022

@author: rggop
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report,confusion_matrix
import transformers
from transformers import AutoModel,BertTokenizerFast,BertModel,AutoModelForSequenceClassification,AutoTokenizer
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


#specify GPU
device = torch.device("cuda")


#specify path to test data 
test_data_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\phrase_polarity_test_data.csv"

#load test data
test_data = pd.read_csv(test_data_path,dtype=str)

test_input = test_data["phrase"].to_list()
test_labels = test_data["polarity"].to_numpy().reshape(-1,1)

#convert labels to onehot vectors
enc = OneHotEncoder(handle_unknown = 'ignore')
enc.fit(test_labels)
test_labels_onehot = enc.transform(test_labels).toarray()

print("Labels assigned")
print(enc.inverse_transform([[1,0,0]]))
print(enc.inverse_transform([[0,1,0]]))
print(enc.inverse_transform([[0,0,1]]))


MAX_LENGTH = 25
#load the BERT tiny tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

#Tokenize and encode sequence in the test set
tokens_test = bert_tokenizer.batch_encode_plus(test_input, max_length = MAX_LENGTH,
                              pad_to_max_length = True, truncation = True,
                              return_token_type_ids= False)

#convert integer tokens to pytorch tensors

#convert test set
test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels_onehot)


#Load saved model
model_path = 'saved_weights.pth'
eval_model = torch.load(model_path)

#get predictions for val data
with torch.no_grad():
    preds_eval = eval_model(test_seq.to(device), test_mask.to(device))[0]
    preds_eval = preds_eval.detach().cpu().numpy()

#evaluate models performance 
preds_eval = np.argmax(preds_eval,axis = 1)
val_y_eval = np.argmax(test_y,axis = 1)
print(classification_report(val_y_eval,preds_eval))

#confusion matrix
print(confusion_matrix(val_y_eval,preds_eval))