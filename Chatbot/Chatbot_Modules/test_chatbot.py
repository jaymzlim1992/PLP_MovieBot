import torch
import time


import utils
import utils_model
import chatbot
import parameter.initial_states as states


# Load Tokenizer & Model
nlu_tokenizer, nlg_tokenizer, nlu_model, nlg_model = utils_model.load_tokenizer_model()

print('>>>>> Round 1')
usr_uttr1 = 'I need some movie recommendations'
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start1 = time.time()
chat1, print1 = chatbot.run_chatbot(usr_uttr1, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat1)
print(time.time()-start1)

print('>>>>> Round 2')
usr_uttr2 = 'A romance movie'
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start2 = time.time()
chat2, print2 = chatbot.run_chatbot(usr_uttr2, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat2)
print(time.time()-start2)

print('>>>>> Round 3')
usr_uttr3 = 'thank you'
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start3 = time.time()
chat3, print3 = chatbot.run_chatbot(usr_uttr3, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat3)
print(time.time()-start3)

print('>>>>> Round 4')
usr_uttr4 = 'thank you'
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start4 = time.time()
chat4, print4 = chatbot.run_chatbot(usr_uttr4, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat4)
print(time.time()-start4)

print('>>>>> Round 5')
usr_uttr5 = "I want to know what people think about a movie"
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start5 = time.time()
chat5, print5 = chatbot.run_chatbot(usr_uttr5, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat5)
print(time.time()-start5)

print('>>>>> Round 6')
usr_uttr6 = "The movie title is Inception"
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start6 = time.time()
chat6, print6 = chatbot.run_chatbot(usr_uttr6, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat6)
print(time.time()-start6)

print('>>>>> Round 7')
usr_uttr7 = "That's ok. That's it for now. Thanks"
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start7 = time.time()
chat7, print7 = chatbot.run_chatbot(usr_uttr7, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat7)
print(time.time()-start7)

print('>>>>> Round 8')
usr_uttr8 = "Thanks, that's helpful"
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start8 = time.time()
chat8, print8 = chatbot.run_chatbot(usr_uttr8, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat8)
print(time.time()-start8)

print('>>>>> Round 9')
usr_uttr9 = "What's the weather today"
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start9 = time.time()
chat9, print9 = chatbot.run_chatbot(usr_uttr9, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat9)
print(time.time()-start9)

print('>>>>> Round 9')
usr_uttr9 = "What's the weather today"
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start9 = time.time()
chat9, print9 = chatbot.run_chatbot(usr_uttr9, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat9)
print(time.time()-start9)


print('>>>>> Round 10')
usr_uttr10 = "Sure, recommend me some movies then"
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start10 = time.time()
chat10, print10 = chatbot.run_chatbot(usr_uttr10, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat10)
print(time.time()-start10)


print('>>>>> Round 11')
usr_uttr11 = "Maybe some sci-fi movies acted by Robert Dsq"
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start11 = time.time()
chat11, print11 = chatbot.run_chatbot(usr_uttr11, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat11)
print(time.time()-start10)


print('>>>>> Round 12')
usr_uttr12 = 'dummy utterance oos intent - what is the current intent'
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start12 = time.time()
chat12, print12 = chatbot.run_chatbot(usr_uttr12, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat12)
print(time.time()-start12)

print('>>>>> Round 13')
usr_uttr13 = "What is everyone talking about The Batman"
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start13 = time.time()
chat13, print13 = chatbot.run_chatbot(usr_uttr13, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat13)
print(time.time()-start13)

print('>>>>> Round 14')
usr_uttr14 = 'dummy utterance oos intent - what is the current intent'
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start14 = time.time()
chat14, print14 = chatbot.run_chatbot(usr_uttr14, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat14)
print(time.time()-start14)

print('>>>>> Round 15')
usr_uttr15 = "Interesting results! that will be it for today."
print(states.active_intent)
print(states.slots)
print(states.chatbot_response)
print(states.sentiment_slot_ready, "|", states.recommender_slot_ready)
print(states.sentiment_api_call, "|", states.recommender_api_call)
start15 = time.time()
chat15, print15 = chatbot.run_chatbot(usr_uttr15, nlu_tokenizer, nlu_model, nlg_tokenizer, nlg_model)
print(chat15)
print(time.time()-start15)


print('='* 150)
print('user1: ', usr_uttr1)
print('sys1:  ', chat1)
print('user2: ', usr_uttr2)
print('sys2:  ', chat2)
# print('user3: ', usr_uttr3)
print('sys3:  ', chat3)
print('user4: ', usr_uttr4)
print('sys4:  ', chat4)
print('user5: ', usr_uttr5)
print('sys5:  ', chat5)
print('user6: ', usr_uttr6)
print('sys6:  ', chat6)
# print('user7: ', usr_uttr7)
print('sys7:  ', chat7)
print('user8: ', usr_uttr8)
print('sys8:  ', chat8)
print('user9: ', usr_uttr9)
print('sys9:  ', chat9)
print('user10: ', usr_uttr10)
print('sys10:  ', chat10)
print('user11: ', usr_uttr11)
print('sys11:  ', chat11)
# print('user12: ', usr_uttr12)
print('sys12:  ', chat12)
print('user13: ', usr_uttr13)
print('sys13:  ', chat13)
# print('user14: ', usr_uttr14)
print('sys14:  ', chat14)
print('user15: ', usr_uttr15)
print('sys15:  ', chat15)


