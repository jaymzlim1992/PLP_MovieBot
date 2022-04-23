# Chatbot NLU & NLG Tokenizer and Model Class

import Chatbot_Modules.config as config
import torch
import torch.nn as nn

from transformers import BertTokenizerFast, BertModel
from transformers import T5Tokenizer, T5ForConditionalGeneration


# ===== Natural Language Understanding (NLU) =====
# 1. NLU Model Architecture:
# Intent Classifier
class IntentClassifier(nn.Module):
    def __init__(self, in_dim, num_intent, dropout_rate=0.0):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_dim, num_intent)

    def forward(self, hidden_states_input):
        output = self.dropout(hidden_states_input)
        output = self.linear(output)

        return output


# Slot Classifier
class SlotClassifier(nn.Module):
    def __init__(self, in_dim, num_slot_label, dropout_rate=0.0):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_dim, num_slot_label)

    def forward(self, hidden_states_input):
        output = self.dropout(hidden_states_input)
        output = self.linear(output)

        return output


# Joint Intent & Slot Model Class
class BERT_Joint_Intent_Slot(nn.Module):
    def __init__(self, model_name, num_intent, num_slot_label, dropout_rate=0.0):
        super(BERT_Joint_Intent_Slot, self).__init__()

        self.encoder = BertModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.dropout_rate = dropout_rate
        self.IntentClassifier = IntentClassifier(self.hidden_size, num_intent, self.dropout_rate)
        self.SlotClassifier = SlotClassifier(self.hidden_size, num_slot_label, self.dropout_rate)

    def forward(self, batch_ids, batch_mask):
        output = self.encoder(input_ids=batch_ids, attention_mask=batch_mask)
        sequence_output = output[0]
        pooled_output = output[1]

        intent_logits = self.IntentClassifier(pooled_output)
        slot_logits = self.SlotClassifier(sequence_output)

        return intent_logits, slot_logits


# 2. Load Tokenizer - BERT Tokenizer from HuggingFace Transformer library
def get_nlu_tokenizer(model_name=config.nlu_model_name):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    return tokenizer


# 3. Load Model - Joint Intent & Slot Classifier based on BERT Architecture (Pre-Trained and Fine-Tuned)
def get_nlu_model(model_name=config.nlu_model_name, num_intent=config.nlu_num_intent,
                  num_slot_label=config.nlu_num_slot, model_state_path=config.nlu_model_path):
    nlu_model = BERT_Joint_Intent_Slot(model_name=model_name, num_intent=num_intent, num_slot_label=num_slot_label)
    load_model_state = torch.load(model_state_path)
    nlu_model.load_state_dict(load_model_state)
    return nlu_model


# ===== Natural Language Generation (NLG) =====
# 1. Load Tokenizer - T5-small from HuggingFace Transformer Library
def get_nlg_tokenizer(model_name=config.nlg_model_name):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return tokenizer


# 2. Load Model - T5-small from HuggingFace Transformer Library (Fine-Tuned)
def get_nlg_model(model_name=config.nlg_model_name, model_state_path=config.nlg_model_path):
    nlg_model = T5ForConditionalGeneration.from_pretrained(model_name)
    load_model_state = torch.load(model_state_path)
    nlg_model.load_state_dict(load_model_state)
    return nlg_model


# ===== Tokenizer and Model Initialization Function =====
# 1. Load tokenizer and models:
def load_tokenizer_model():
    nlu_tokenizer = get_nlu_tokenizer()
    nlg_tokenizer = get_nlg_tokenizer()
    nlu_model = get_nlu_model()
    nlg_model = get_nlg_model()
    return nlu_tokenizer, nlg_tokenizer, nlu_model, nlg_model
