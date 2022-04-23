import torch
import time
import Chatbot_Modules.utils as utils
import Chatbot_Modules.utils_model as utils_model
import Chatbot_Modules.parameter.initial_states as states


def run_chatbot(user_utterance, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model,
                sent_model,sent_tokenizer,QA_model):
    current_intent = utils.update_dialog_states(user_utterance, nlu_tokenizer, nlu_model,
                                                sent_model,sent_tokenizer,QA_model)
    text_response, print_result = utils.get_sys_response(current_intent, nlg_tokenizer, nlg_model)
    return text_response, print_result


