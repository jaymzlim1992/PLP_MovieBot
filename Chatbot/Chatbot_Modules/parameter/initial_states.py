# Initial States for Chatbot
# This file list all initial variables which the system will read

"""
Active Intent (Follows Intent Mapping)
3 - Movie Sentiment
4 - Movie Recommendations
"""
active_intent = None


"""
Slot Values for both Movie Sentiment and Recommendation API
"""
slots = {'title': None, 'genre': None, 'actor': None, 'director': None}

"""
List to store Movie Recomendation Results
"""
movie_recommendation_result = []

"""
Flag for chatbot to continue printing responses
0 - wait for user utterance
1 - continue running chatbot responses
"""
chatbot_response = 0

"""
Flag for whether the slots for sentiment and recommender has been filled sufficiently
0 - Not filled
1 - Filled sufficiently
"""
sentiment_slot_ready = 0
recommender_slot_ready = 0


"""
Flag for whether sentiment or recommender API has been called once
0 - Not run yet
1 - Success
2 - Failure
"""
sentiment_api_call = 0
recommender_api_call = 0

"""
Flag for requesting further action for API failure to give results cases
0 - No
1 - Request More
"""
sentiment_failure_reqmore = 0
recommender_failure_reqmore = 0


"""
Flag for whether sentiment or recommender task has been completed
0 - Not run yet
1 - Completed (Regardless success or failure)
"""
sentiment_complete = 0
recommender_complete = 0


