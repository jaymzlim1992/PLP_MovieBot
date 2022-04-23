# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 19:10:00 2022

@author: rggop
"""
#import all packages

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report,confusion_matrix
import transformers
from transformers import AutoModel,BertTokenizerFast,BertModel,AutoModelForSequenceClassification,AutoTokenizer
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
from transformers import AdamW
import matplotlib.pyplot as plt
#specify GPU
device = torch.device("cuda")


#specify path to train and val data 
train_data_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\train_data.csv"
val_data_path = "D:\\PLP_Project\\Sentiment Analysis\\dataset\\val_data.csv"

#load train and validation data
train_data = pd.read_csv(train_data_path,dtype=str)
val_data = pd.read_csv(val_data_path,dtype=str)

train_input = train_data["phrase"].to_list()
train_labels = train_data["polarity"].to_numpy().reshape(-1,1)
val_input = val_data["phrase"].to_list()
val_labels = val_data["polarity"].to_numpy().reshape(-1,1)

#convert labels to onehot vectors
enc = OneHotEncoder(handle_unknown = 'ignore')
enc.fit(train_labels)
train_labels_onehot = enc.transform(train_labels).toarray()
val_labels_onehot = enc.transform(val_labels).toarray()

print("Labels assigned")
print(enc.inverse_transform([[1,0,0]]))
print(enc.inverse_transform([[0,1,0]]))
print(enc.inverse_transform([[0,0,1]]))

#check class distribution in train and val data
print("Train Distribution")
print(train_data["polarity"].value_counts(normalize = True))
print("Val Distribution")
print(val_data["polarity"].value_counts(normalize = True))

#Find the max Lenght of BERT input sequence
#seq_len = [len(i.split()) for i in train_data["phrase"]]
#pd.Series(seq_len).hist(bins = 30)
MAX_LENGTH = 25


#import BERT tiny pretrained model
#bert_tiny = BertModel.from_pretrained("bert-base-uncased",return_dict=False)
bert_tiny = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base",
            num_labels=3)#,output_attentions = False,output_hidden_states = False)
#load the BERT tiny tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")


#Tokenize and encode sequence in the training set
tokens_train = bert_tokenizer.batch_encode_plus(train_input, max_length = MAX_LENGTH,
                              pad_to_max_length = True, truncation = True,
                              return_token_type_ids= False)
#Tokenize and encode sequence in the val set
tokens_val = bert_tokenizer.batch_encode_plus(val_input, max_length = MAX_LENGTH,
                              pad_to_max_length = True, truncation = True,
                              return_token_type_ids= False)

#convert integer tokens to pytorch tensors

#convert train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels_onehot)
#convert val set
val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels_onehot)


#Define Data Loaders for train set
#define batch size
batch_size = 16
#combine tensors 
train_dataset = TensorDataset(train_seq,train_mask,train_y)
#sampler for sampling the data during training
train_sampler = RandomSampler(train_dataset)
#data loader for train set
train_dataloader = DataLoader(train_dataset,sampler=train_sampler,batch_size= batch_size)


#Define Data Loaders for val set
#combine tensors 
val_dataset = TensorDataset(val_seq,val_mask,val_y)
#sampler for sampling the data during training
val_sampler = RandomSampler(val_dataset)
#data loader for train set
val_dataloader = DataLoader(val_dataset,sampler=val_sampler,batch_size= batch_size)


#pass pre trained bert to define our architecture
#model = Senti_Arch(bert_tiny)
model = bert_tiny
#pass model to GPU
model = model.to(device)


#to handle class imbalance in data calcualte class weights 
class_wts = compute_class_weight(class_weight= 'balanced',classes = np.unique(np.argmax(train_labels_onehot, axis=1).flatten()),
                                 y = np.argmax(train_labels_onehot, axis=1).flatten())
#class_wts = [1.0,1.0,1.0]
print("class weights :",class_wts)        
#convert class weights to tensors
weights = torch.tensor(class_wts, dtype= torch.float)
weights = weights.to(device)
        
#Define Loss function
categorical_cross_entropy = nn.CrossEntropyLoss(weight=weights)

#Define Optimizer 
optimizer = AdamW(model.parameters(), lr = 5e-7)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

#Define Function to train the model

def train():
    model.train()
    
    total_loss,total_accuracy = 0,0
    
    #empty list to save model predictions
    total_preds = []
    
    #iterate over batches
    for step,batch in enumerate(train_dataloader):
        #progress update after every 100 batches
        if step % 100 == 0 and step !=0:
            print(' Batch  {:5>,} of {:5,}.'.format(step,len(train_dataloader)))
            
        #push the batch to GPU
        batch = [r.to(device) for r in batch]
        
        sent_id, mask, labels = batch
        
        #clear previously calculated gradients
        model.zero_grad()
        
        #get model prediction for current batch
        preds = model(sent_id,token_type_ids=None,attention_mask=mask)[0]
        
        #compute loss between prediction and groundtruth
        loss = categorical_cross_entropy(preds,labels)
        #loss,preds = model(sent_id,token_type_ids=None,attention_mask=mask,
        #                    labels= labels)[:2]
        #add on to the total loss 
        total_loss = total_loss + loss.item()
        #model predictions are stored on GPU so push it to CPU
        preds = preds.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        # accumulate it over all batches.
        total_accuracy = total_accuracy + flat_accuracy(preds, labels)
        #backward pass to calculate gradients
        loss.backward()
        
        #clip the gradients to 1 : this will help to prevent exploding
        #                          gradient problem
        torch.nn.utils.clip_grad_norm(model.parameters(),1.0)
        
        #update parameters
        optimizer.step()
        
        #model predictions are stored on GPU so push it to CPU
        #preds = preds.detach().cpu().numpy()
        
        #append the model predictions
        total_preds.append(preds)
        
    #compute the training loss of the epoch
    avg_loss = total_loss/len(train_dataloader)
    #compute train accuracy 
    avg_acc = total_accuracy/len(train_dataloader)
    #total predictions are in the form of (no: batches, size of batch, no: of classes)
    #reshape the predictions in the form (number of samples , no: of classes)
    total_preds = np.concatenate(total_preds, axis = 0)
    
    #return the loss and predictions
    return avg_loss,avg_acc, total_preds 

#Function to evaluate the model
def evaluate():
    print("\nEvaluating....")
    
    #deactivate dropout layers
    model.eval()
    
    total_loss,total_accuracy = 0,0
    total_eval_accuracy = 0
    #empty list to save model predictions
    total_preds = []
    
    #iterate over batches
    for step,batch in enumerate(val_dataloader):
        #progress update after every 25 batches
        if step % 25 == 0 and step !=0:       
            # calculate the elapsed time in minutes
            #elapsed = format_time(time.time() - t0)
            #report progress
            print(' Batch  {:5>,} of {:5,}.'.format(step,len(val_dataloader)))
        #push the batch to GPU
        batch = [r.to(device) for r in batch]
        
        sent_id, mask, labels = batch
        
        #deactivate auto grad
        with torch.no_grad():        
            #get model prediction for current batch
            #preds = model(sent_id, mask)
        
            #compute loss between prediction and groundtruth
            #loss = categorical_cross_entropy(preds,labels)
            loss,preds = model(sent_id,token_type_ids=None,attention_mask=mask,
                                labels= labels)[:2]
            #add on to the total loss 
            total_loss = total_loss + loss.item()
        
            #model predictions are stored on GPU so push it to CPU
            preds = preds.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            
            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy = total_eval_accuracy + flat_accuracy(preds, labels)    
            #append the model predictions
            total_preds.append(preds)   
    #compute the validation loss
    avg_loss = total_loss/len(val_dataloader)
    #compute validation accuracy 
    avg_acc = total_eval_accuracy/len(val_dataloader)
    #total predictions are in the form of (no: batches, size of batch, no: of classes)
    #reshape the predictions in the form (number of samples , no: of classes)
    total_preds = np.concatenate(total_preds, axis = 0)
    
    
    
    #return the loss and predict
    return avg_loss, avg_acc,total_preds 



#START MODEL TRAINING

NO_EPOCHS = 25
#set initial loss to infinite
best_val_loss = float('inf')

#empty list to store train and val loss of every epoch
train_losses = []
val_losses = []
train_accuracy = []
val_accuracy = []
#for each epoch
for epoch in range(NO_EPOCHS):
    print('\n Epochs {:} / {:}'.format(epoch + 1,NO_EPOCHS))
    
    #train model
    train_loss,train_acc,_ = train()
    
    #val model
    val_loss,val_acc,_ = evaluate()
    
    #save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model,'saved_weights.pth')
    
    #append training and validation loss
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    #append training and validation accuracy
    train_accuracy.append(train_acc)
    val_accuracy.append(val_acc)
    #print loss
    print(f'Train Loss :  {train_loss:.3f}')
    print(f'Validation Loss :  {val_loss:.3f}')
    print(f'Validation Accuracy :  {val_acc:.3f}')

#plot loss and accuracy
plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(val_losses,label="val")
plt.plot(train_losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('loss.jpg')
plt.show()


plt.figure(figsize=(10,5))
plt.title("Training and Validation Accuracy")
plt.plot(val_accuracy,label="val")
plt.plot(train_accuracy,label="train")
plt.xlabel("iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('acc.jpg')
plt.show()
    
#Evaluation

#Load saved model
model_path = 'saved_weights.pth'
eval_model = torch.load(model_path)

#get predictions for val data
with torch.no_grad():
    preds_eval = eval_model(val_seq.to(device), val_mask.to(device))[0]
    preds_eval = preds_eval.detach().cpu().numpy()

#evaluate models performance 
preds_eval = np.argmax(preds_eval,axis = 1)
val_y_eval = np.argmax(val_y,axis = 1)
print(classification_report(val_y_eval,preds_eval))

#confusion matrix
print(confusion_matrix(val_y_eval,preds_eval))


#get predictions for train data
with torch.no_grad():
    preds_eval = eval_model(train_seq.to(device), train_mask.to(device))[0]
    preds_eval = preds_eval.detach().cpu().numpy()

#evaluate models performance 
preds_eval = np.argmax(preds_eval,axis = 1)
train_y_eval = np.argmax(train_y,axis = 1)
print(classification_report(train_y_eval,preds_eval))

#confusion matrix
print(confusion_matrix(train_y_eval,preds_eval))

