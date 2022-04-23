import os

# BASE DIRECTORY
basedir = os.path.abspath(os.path.dirname(__file__))

# CHATBOT CONFIGURATIONS
# ===== Natural Language Understanding (NLU) =====
nlu_model_path = 'Models\\nlu_nlg_models\\bert_nlu.pt'
nlu_model_name = 'bert-base-uncased'
nlu_num_intent = 8
nlu_num_slot = 9
nlu_excluded_ids = [101, 102, 0]
nlu_token_length = 32

# ===== Natural Language Generation (NLG) =====
nlg_model_path = 'Models\\nlu_nlg_models\\t5_nlg.pt'
nlg_model_name = 't5-small'
nlg_token_length = 64
nlg_beam_size = 5

