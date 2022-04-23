import torch
import Chatbot_Modules.config as config
import numpy as np
import Chatbot_Modules.parameter.initial_states as states
import Chatbot_Modules.parameter.system_control_code as syscode
from Sentiment_Analysis.aspect_sentiment_pipeline_v2 import aspect_sentiment_analysis
from Movie_Recomendation.movie_recomendation_pipeline import recommender


# Utility functions for Chatbot
# ===== Natural Language Understanding (NLU) =====
# 1. Get Slot Values of Specific Type
def extract_slot_values(user_utterance, encoded_bio, offset_mapping, slot_b_index, slot_i_index):
    slot_value = []
    start_char = None
    for i in range(len(encoded_bio)):
        if encoded_bio[i] == slot_b_index:
            if i == len(encoded_bio) - 1:
                slot_value.append(user_utterance[offset_mapping[i][0]:offset_mapping[i][1]])
            elif encoded_bio[i + 1] != slot_i_index:
                slot_value.append(user_utterance[offset_mapping[i][0]:offset_mapping[i][1]])
            else:
                start_char = offset_mapping[i][0]
        if encoded_bio[i] == slot_i_index:
            if i == len(encoded_bio) - 1:
                end_char = offset_mapping[i][1]
                slot_value.append(user_utterance[start_char: end_char])
                start_char = None
            elif encoded_bio[i + 1] != slot_i_index:
                end_char = offset_mapping[i][1]
                slot_value.append(user_utterance[start_char: end_char])
                start_char = None
    return slot_value


# 2. Generate Encoded Intent and BIO Tags
def get_usr_nlu(user_utterance, nlu_tokenizer, nlu_model, utterance_maxlength=config.nlu_token_length,
                excluded_ids=config.nlu_excluded_ids):
    slots = {'title': None, 'genre': None, 'actor': None, 'director': None}
    nlu_model.eval()
    with torch.no_grad():
        tokenized_inputs = nlu_tokenizer(user_utterance, padding='max_length', max_length=utterance_maxlength,
                                         truncation=True, return_attention_mask=True, return_offsets_mapping=True,
                                         return_tensors='pt')
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        offset_mapping = tokenized_inputs['offset_mapping']
        token_mask = np.asarray([x not in excluded_ids for x in input_ids.numpy()[0]])

        intent_logits, slot_logits = nlu_model(input_ids, attention_mask)

        encoded_intent = intent_logits[0].numpy().argmax(axis=-1)
        encoded_bio = slot_logits[0].numpy().argmax(axis=-1)[token_mask]
        offset_mapping_masked = offset_mapping[0][token_mask]
        actor_found = extract_slot_values(user_utterance, encoded_bio, offset_mapping_masked, 0, 4)
        director_found = extract_slot_values(user_utterance, encoded_bio, offset_mapping_masked, 1, 5)
        genre_found = extract_slot_values(user_utterance, encoded_bio, offset_mapping_masked, 2, 6)
        title_found = extract_slot_values(user_utterance, encoded_bio, offset_mapping_masked, 3, 7)

        if len(title_found) > 0:
            slots['title'] = title_found
        if len(genre_found) > 0:
            slots['genre'] = genre_found
        if len(actor_found) > 0:
            slots['actor'] = actor_found
        if len(director_found) > 0:
            slots['director'] = director_found

    return encoded_intent, slots


# ===== Natural Language Generation (NLG) =====
# 1. Generate Text Response Given Structured Inputs
def get_sys_nlg(control_code_input, nlg_tokenizer, nlg_model, generation_length=config.nlg_token_length,
                beam_size=config.nlg_beam_size):
    nlg_model.eval()
    with torch.no_grad():
        input_ids = nlg_tokenizer.encode(control_code_input, padding='max_length', max_length=generation_length,
                                         truncation=True, return_tensors='pt')
        generated_ids = nlg_model.generate(input_ids=input_ids, max_length=generation_length, num_beams=beam_size)
        generated_text = nlg_tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text


# ===== Dialogue States Updates =====
# Rule-Based to update dialogue states
def update_dialog_states(user_utterance, nlu_tokenizer, nlu_model,sent_model, 
                         sent_tokenizer, QA_model):
    usr_intent, usr_slots = get_usr_nlu(user_utterance, nlu_tokenizer, nlu_model)
    
    # Condition for continuous system response generation (no usr_intent inference required)
    if states.chatbot_response == 1:
        usr_intent = None

    # Inform Intent - Sentiment + Slot filling if found
    if usr_intent == 4 and states.active_intent is None:
        states.active_intent = 4
        if usr_slots['title'] is not None:
            states.slots['title'] = usr_slots['title'][0]
            states.sentiment_slot_ready = 1
            states.chatbot_response = 1

    # Inform Intent - Recommend + Slot filling if found
    elif usr_intent == 3 and states.active_intent is None:
        states.active_intent = 3
        if usr_slots['genre'] is not None:
            states.slots['genre'] = usr_slots['genre'][0]
            states.recommender_slot_ready = 1
            states.chatbot_response = 1
        if usr_slots['director'] is not None:
            states.slots['director'] = usr_slots['director'][0]
            states.recommender_slot_ready = 1
            states.chatbot_response = 1
        if usr_slots['actor'] is not None:
            states.slots['actor'] = usr_slots['actor'][0]
            states.recommender_slot_ready = 1
            states.chatbot_response = 1

    # Inform Sentiment Slot filling
    elif usr_intent == 2 and states.active_intent == 4:
        if usr_slots['title'] is not None:
            states.slots['title'] = usr_slots['title'][0]
            states.sentiment_slot_ready = 1
            states.chatbot_response = 1

    # Inform Recommender Slot filling
    elif usr_intent == 2 and states.active_intent == 3:
        if usr_slots['genre'] is not None:
            states.slots['genre'] = usr_slots['genre'][0]
            states.recommender_slot_ready = 1
            states.chatbot_response = 1
        if usr_slots['director'] is not None:
            states.slots['director'] = usr_slots['director'][0]
            states.recommender_slot_ready = 1
            states.chatbot_response = 1
        if usr_slots['actor'] is not None:
            states.slots['actor'] = usr_slots['actor'][0]
            states.recommender_slot_ready = 1
            states.chatbot_response = 1

    # Run Sentiment API after system informed searching
    elif states.active_intent == 4 and states.sentiment_slot_ready == 1 and states.sentiment_api_call == 0 and \
            states.sentiment_failure_reqmore == 0:
        # Run sentiment API
        states.sentiment_api_call = aspect_sentiment_analysis(states.slots['title'],
                                                              sent_model, sent_tokenizer,
                                                              QA_model) 
        # Update APi results
        if states.sentiment_api_call == 2:
            states.chatbot_response = 1
            states.sentiment_failure_reqmore = 1
        else:
            states.chatbot_response = 0

    # Run Recommender API after system informed searching
    elif states.active_intent == 3 and states.recommender_slot_ready == 1 and states.recommender_api_call == 0 and \
            states.recommender_failure_reqmore == 0:
        # Run recommender API
        states.movie_recommendation_result = recommender(states.slots['genre'],
                                                         states.slots['director'],
                                                         states.slots['actor'])
        states.recommender_api_call = 1
        
        # Update APi results
        # Update APi results
        if states.recommender_api_call == 2:
            states.chatbot_response = 1
            states.recommender_failure_reqmore = 1
        else:
            states.chatbot_response = 0

    # # Reset failure result back to normal (chatbot will wait for user utterance/actions next) - Sentiment
    # elif states.active_intent == 4 and states.sentiment_slot_ready == 1 and states.sentiment_api_call == 1 and \
    #         states.sentiment_failure_reqmore == 1:
    #     states.chatbot_response = 0
    #     states.sentiment_failure_reqmore = 0
    #
    # # Reset failure result back to normal (chatbot will wait for user utterance/actions next) - Recommender
    # elif states.active_intent == 4 and states.recommender_slot_ready == 1 and states.recommender_api_call == 1 and \
    #         states.sentiment_failure_reqmore == 1:
    #     states.chatbot_response = 0
    #     states.sentiment_failure_reqmore = 0

    # Update completed sentiment task
    elif states.active_intent == 4 and states.sentiment_slot_ready == 1 and states.sentiment_api_call != 0 and \
            states.sentiment_complete == 0:
        states.sentiment_complete = 1
        states.sentiment_failure_reqmore = 0
        states.chatbot_response = 0
        
    # Update recommender sentiment task
    elif states.active_intent == 3 and states.recommender_slot_ready == 1 and states.recommender_api_call != 0 and \
            states.recommender_complete == 0:
        states.recommender_complete = 1
        states.recommender_failure_reqmore = 0
        states.chatbot_response = 0
        
    # Reset states if user changing active_intent - sentiment
    if usr_intent == 4 and states.active_intent is not None:
        states.active_intent = 4
        states.sentiment_slot_ready = 0
        states.sentiment_api_call = 0
        states.sentiment_complete = 0
        states.recommender_slot_ready = 0
        states.recommender_api_call = 0
        states.recommender_complete = 0
        states.movie_recommendation_result = []
        states.slots = {'title': None, 'genre': None, 'actor': None, 'director': None}
        if usr_slots['title'] is not None:
            states.slots['title'] = usr_slots['title'][0]
            states.sentiment_slot_ready = 1
            states.chatbot_response = 1

    # Reset states if user changing active_intent - recommender
    elif usr_intent == 3 and states.active_intent is not None:
        states.active_intent = 3
        states.sentiment_slot_ready = 0
        states.sentiment_api_call = 0
        states.sentiment_complete = 0
        states.recommender_slot_ready = 0
        states.recommender_api_call = 0
        states.recommender_complete = 0
        states.movie_recommendation_result = []
        states.slots = {'title': None, 'genre': None, 'actor': None, 'director': None}
        if usr_slots['genre'] is not None:
            states.slots['genre'] = usr_slots['genre'][0]
            states.recommender_slot_ready = 1
            states.chatbot_response = 1
        if usr_slots['director'] is not None:
            states.slots['director'] = usr_slots['director'][0]
            states.recommender_slot_ready = 1
            states.chatbot_response = 1
        if usr_slots['actor'] is not None:
            states.slots['actor'] = usr_slots['actor'][0]
            states.recommender_slot_ready = 1
            states.chatbot_response = 1
    print("##############################")
    print("current_intent: ", usr_intent)
    print("states.active_intent: ", states.active_intent)
    print("states.sentiment_slot_ready :", states.sentiment_slot_ready)
    print("states.chatbot_response :", states.chatbot_response)
    print("states.sentiment_api_call :", states.sentiment_api_call)
    print("states.sentiment_failure_reqmore :", states.sentiment_failure_reqmore)
    print("states.sentiment_complete :", states.sentiment_complete)
    return usr_intent


# ===== Dialog Policy (System Actions) =====
# Rule-Based to decide system actions
def get_sys_response(current_intent, nlg_tokenizer, nlg_model):
    control_code = None
    print_result = 0

    # ----------------------------------- Policy for requesting slot values -----------------------------------
    # System Request for slot values - Sentiment
    if states.active_intent == 4 and states.sentiment_slot_ready == 0 and states.sentiment_api_call == 0 and \
            states.sentiment_complete == 0:
        control_code = syscode.sentiment_slot_request

    # System Request for slot values - Recommender
    elif states.active_intent == 3 and states.recommender_slot_ready == 0 and states.recommender_api_call == 0 and \
            states.recommender_complete == 0:
        control_code = syscode.recommend_slot_request

    # ------------------------------------ Policy for informing searching -------------------------------------
    # System Inform Search - Sentiment
    elif states.active_intent == 4 and states.sentiment_slot_ready == 1 and states.sentiment_api_call == 0 and \
            states.sentiment_complete == 0:
        control_code = syscode.sentiment_searching
        control_code = control_code.replace('var1', states.slots['title'])

    # System Inform Search - Recommender
    elif states.active_intent == 3 and states.recommender_slot_ready == 1 and states.recommender_api_call == 0 and \
            states.recommender_complete == 0:
        genre_slot = states.slots['genre']
        director_slot = states.slots['director']
        actor_slot = states.slots['actor']

        if genre_slot is not None and director_slot is None and actor_slot is None:
            control_code = syscode.recommend_searching_genre
            control_code = control_code.replace('var1', genre_slot)

        elif genre_slot is None and director_slot is not None and actor_slot is None:
            control_code = syscode.recommend_searching_director
            control_code = control_code.replace('var1', director_slot)

        elif genre_slot is None and director_slot is None and actor_slot is not None:
            control_code = syscode.recommend_searching_actor
            control_code = control_code.replace('var1', actor_slot)

        elif genre_slot is not None and director_slot is not None and actor_slot is None:
            control_code = syscode.recommend_searching_genre_director
            control_code = control_code.replace('var1', genre_slot)
            control_code = control_code.replace('var2', director_slot)

        elif genre_slot is not None and director_slot is None and actor_slot is not None:
            control_code = syscode.recommend_searching_genre_actor
            control_code = control_code.replace('var1', genre_slot)
            control_code = control_code.replace('var2', actor_slot)

        elif genre_slot is None and director_slot is not None and actor_slot is not None:
            control_code = syscode.recommend_searching_director_actor
            control_code = control_code.replace('var1', director_slot)
            control_code = control_code.replace('var2', actor_slot)

        elif genre_slot is not None and director_slot is not None and actor_slot is not None:
            control_code = syscode.recommend_searching_genre_director_actor
            control_code = control_code.replace('var1', genre_slot)
            control_code = control_code.replace('var2', director_slot)
            control_code = control_code.replace('var3', actor_slot)

    # -------------------------------------- Policy for Search Success ----------------------------------------
    # System Inform Search Success - Sentiment
    elif states.active_intent == 4 and states.sentiment_slot_ready == 1 and states.sentiment_api_call == 1 and \
            states.sentiment_complete == 0:
        control_code = syscode.sentiment_success
        control_code = control_code.replace('var1', states.slots['title'])
        print_result = 1

    # System Inform Search Success - Recommender
    elif states.active_intent == 3 and states.recommender_slot_ready == 1 and states.recommender_api_call == 1 and \
            states.recommender_complete == 0:
        genre_slot = states.slots['genre']
        director_slot = states.slots['director']
        actor_slot = states.slots['actor']

        if genre_slot is not None and director_slot is None and actor_slot is None:
            control_code = syscode.recommend_success_genre
            control_code = control_code.replace('var1', genre_slot)

        elif genre_slot is None and director_slot is not None and actor_slot is None:
            control_code = syscode.recommend_success_director
            control_code = control_code.replace('var1', director_slot)

        elif genre_slot is None and director_slot is None and actor_slot is not None:
            control_code = syscode.recommend_success_actor
            control_code = control_code.replace('var1', actor_slot)

        elif genre_slot is not None and director_slot is not None and actor_slot is None:
            control_code = syscode.recommend_success_genre_director
            control_code = control_code.replace('var1', genre_slot)
            control_code = control_code.replace('var2', director_slot)

        elif genre_slot is not None and director_slot is None and actor_slot is not None:
            control_code = syscode.recommend_success_genre_actor
            control_code = control_code.replace('var1', genre_slot)
            control_code = control_code.replace('var2', actor_slot)

        elif genre_slot is None and director_slot is not None and actor_slot is not None:
            control_code = syscode.recommend_success_director_actor
            control_code = control_code.replace('var1', director_slot)
            control_code = control_code.replace('var2', actor_slot)

        elif genre_slot is not None and director_slot is not None and actor_slot is not None:
            control_code = syscode.recommend_success_genre_director_actor
            control_code = control_code.replace('var1', genre_slot)
            control_code = control_code.replace('var2', director_slot)
            control_code = control_code.replace('var3', actor_slot)

        print_result = 2

    # -------------------------------------- Policy for Search Failure ----------------------------------------
    # System Inform Search Failure - Sentiment
    elif states.active_intent == 4 and states.sentiment_slot_ready == 1 and states.sentiment_api_call == 2 and \
            states.sentiment_complete == 0 and states.sentiment_failure_reqmore == 1:
        control_code = syscode.sentiment_failure
        control_code = control_code.replace('var1', states.slots['title'])

    # System Inform Search Failure - Recommender
    elif states.active_intent == 3 and states.recommender_slot_ready == 1 and states.recommender_api_call == 2 and \
            states.recommender_complete == 0 and states.recommender_failure_reqmore == 1:
        genre_slot = states.slots['genre']
        director_slot = states.slots['director']
        actor_slot = states.slots['actor']

        if genre_slot is not None and director_slot is None and actor_slot is None:
            control_code = syscode.recommend_failure_genre
            control_code = control_code.replace('var1', genre_slot)

        elif genre_slot is None and director_slot is not None and actor_slot is None:
            control_code = syscode.recommend_failure_director
            control_code = control_code.replace('var1', director_slot)

        elif genre_slot is None and director_slot is None and actor_slot is not None:
            control_code = syscode.recommend_failure_actor
            control_code = control_code.replace('var1', actor_slot)

        elif genre_slot is not None and director_slot is not None and actor_slot is None:
            control_code = syscode.recommend_failure_genre_director
            control_code = control_code.replace('var1', genre_slot)
            control_code = control_code.replace('var2', director_slot)

        elif genre_slot is not None and director_slot is None and actor_slot is not None:
            control_code = syscode.recommend_failure_genre_actor
            control_code = control_code.replace('var1', genre_slot)
            control_code = control_code.replace('var2', actor_slot)

        elif genre_slot is None and director_slot is not None and actor_slot is not None:
            control_code = syscode.recommend_failure_director_actor
            control_code = control_code.replace('var1', director_slot)
            control_code = control_code.replace('var2', actor_slot)

        elif genre_slot is not None and director_slot is not None and actor_slot is not None:
            control_code = syscode.recommend_failure_genre_director_actor
            control_code = control_code.replace('var1', genre_slot)
            control_code = control_code.replace('var2', director_slot)
            control_code = control_code.replace('var3', actor_slot)

    # -------------------------- Policy for Search Failure -> Request Further Actions -------------------------
    # Request for further action from user after search failure - Sentiment
    elif states.active_intent == 4 and states.sentiment_slot_ready == 1 and states.sentiment_api_call == 2 and \
            states.sentiment_failure_reqmore == 0:
        control_code = syscode.sentiment_reqmore

    # Request for further action from user after search failure - Recommender
    elif states.active_intent == 3 and states.recommender_slot_ready == 1 and states.recommender_api_call == 2 and \
            states.recommender_failure_reqmore == 0:
        control_code = syscode.recommend_reqmore

    # --------------------------------- Policy for Ask for Further Actions ------------------------------------
    elif states.active_intent == 4 and current_intent == 7:
        control_code = syscode.sentiment_reqmore

    elif states.active_intent == 3 and current_intent == 7:
        control_code = syscode.recommend_reqmore

    # ------------------------------------- Policy for Closing Goodbye ----------------------------------------
    elif states.active_intent == 4 and current_intent == 0:
        control_code = syscode.sentiment_goodbye

    elif states.active_intent == 3 and current_intent == 0:
        control_code = syscode.recommend_goodbye

    # --------------------------------- Policy for Out-of-Scope (OOS) Intent ----------------------------------
    if current_intent == 6 and \
            (states.active_intent is None or states.sentiment_complete == 1 or states.recommender_complete == 1):
        sys_response = syscode.oos_response_no_active_intent
    elif current_intent == 6 and states.active_intent is not None:
        sys_response = syscode.oos_response_with_active_intent
    elif control_code is None:
        sys_response = syscode.oos_response_with_active_intent
    else:
        sys_response = get_sys_nlg(control_code, nlg_tokenizer, nlg_model)

    return sys_response, print_result
