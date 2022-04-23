# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 10:57:28 2022

@author: rggop
"""
from flask import Flask, render_template, request,jsonify
from Sentiment_Analysis.aspect_sentiment_pipeline_v2 import init_models
import Chatbot_Modules.config as config
import Chatbot_Modules.utils as utils
import Chatbot_Modules.utils_model as utils_model
import Chatbot_Modules.chatbot as chatbot
import Chatbot_Modules.parameter.initial_states as states

app = Flask(__name__)
app.static_folder = 'static'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] =0
#sentiment model path
sentiment_model_path = r"Models\Sentiment_Analysis\saved_weights.pth"
#Flags
FLAG_NOTIFY = 0
#init models
sent_model, sent_tokenizer,QA_model = init_models(sentiment_model_path)
nlu_tokenizer, nlg_tokenizer, nlu_model, nlg_model = utils_model.load_tokenizer_model()


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate,post-check=0,pre-check=0,max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "-1"
    #r.headers['Cache-Control'] = 'public, max-age=0'
    return r
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get')
def get_bot_response():
    #get user response
    user_text = request.args.get('msg')
    #run chatbot 
    response, resp_flag = chatbot.run_chatbot(user_text, nlu_tokenizer, nlu_model, 
                                           nlg_tokenizer, nlg_model,sent_model, 
                                           sent_tokenizer,QA_model)
    #get chat bot flow control flag
    chatbot_response = states.chatbot_response
    #get recommendor results
    if resp_flag == 2:
        recommendor_response = states.movie_recommendation_result
        recommendor_response.insert(0,response)
        response = recommendor_response
    response = jsonify({"message" :response,"print_flag" : resp_flag,
                        "control_flag":chatbot_response})
    return response

if __name__ =="__main__":
    app.run(host="localhost", port=8000)