# System Action Control Codes for NLG
"""
<var1> , <var2> , <var3> to be replaced with respective slots value extracted from NLU
"""


# Control Code for Requesting Slot Values
sentiment_slot_request = 'MOVIE_SENTIMENT REQUEST ( title = ? )'
recommend_slot_request = 'MOVIE_RECOMMEND REQUEST ( genre = ? ; directed_by = ? ; starring = ? )'

# Control Code for Notifying Search
sentiment_searching = 'MOVIE_SENTIMENT NOTIFY_SEARCHING ( title = var1 )'
recommend_searching_genre = 'MOVIE_RECOMMEND NOTIFY_SEARCHING ( genre = var1 )'
recommend_searching_director = 'MOVIE_RECOMMEND NOTIFY_SEARCHING ( director = var1 )'
recommend_searching_actor = 'MOVIE_RECOMMEND NOTIFY_SEARCHING ( actor = var1 )'
recommend_searching_genre_director = 'MOVIE_RECOMMEND NOTIFY_SEARCHING ( genre = var1 ; director = var2 )'
recommend_searching_genre_actor = 'MOVIE_RECOMMEND NOTIFY_SEARCHING ( genre = var1 ; actor = var2 )'
recommend_searching_director_actor = 'MOVIE_RECOMMEND NOTIFY_SEARCHING ( director = var1 ; actor = var2 )'
recommend_searching_genre_director_actor = 'MOVIE_RECOMMEND NOTIFY_SEARCHING ( genre = var1 ; director = var2 ; ' \
                                           'actor = var3 )'

# Control Code for Notifying Success Search
sentiment_success = 'MOVIE_SENTIMENT NOTIFY_SUCCESS ( title = var1 )'
recommend_success_genre = 'MOVIE_RECOMMEND NOTIFY_SUCCESS ( genre = var1 )'
recommend_success_director = 'MOVIE_RECOMMEND NOTIFY_SUCCESS ( director = var1 )'
recommend_success_actor = 'MOVIE_RECOMMEND NOTIFY_SUCCESS ( actor = var1 )'
recommend_success_genre_director = 'MOVIE_RECOMMEND NOTIFY_SUCCESS ( genre = var1 ; director = var2 )'
recommend_success_genre_actor = 'MOVIE_RECOMMEND NOTIFY_SUCCESS ( genre = var1 ; actor = var2 )'
recommend_success_director_actor = 'MOVIE_RECOMMEND NOTIFY_SUCCESS ( director = var1 ; actor = var2 )'
recommend_success_genre_director_actor = 'MOVIE_RECOMMEND NOTIFY_SUCCESS ( genre = var1 ; director = var2 ; ' \
                                         'actor = var3 )'

# Control Code for Notifying Failed Search
sentiment_failure = 'MOVIE_SENTIMENT NOTIFY_FAILURE ( title = var1 )'
recommend_failure_genre = 'MOVIE_RECOMMEND NOTIFY_FAILURE ( genre = var1 )'
recommend_failure_director = 'MOVIE_RECOMMEND NOTIFY_FAILURE ( director = var1 )'
recommend_failure_actor = 'MOVIE_RECOMMEND NOTIFY_FAILURE ( actor = var1 )'
recommend_failure_genre_director = 'MOVIE_RECOMMEND NOTIFY_FAILURE ( genre = var1 ; director = var2 )'
recommend_failure_genre_actor = 'MOVIE_RECOMMEND NOTIFY_FAILURE ( genre = var1 ; actor = var2 )'
recommend_failure_director_actor = 'MOVIE_RECOMMEND NOTIFY_FAILURE ( director = var1 ; actor = var2 )'
recommend_failure_genre_director_actor = 'MOVIE_RECOMMEND NOTIFY_FAILURE ( genre = var1 ; director = var2 ; ' \
                                         'actor = var3 )'

# Control Code to offer further help
sentiment_reqmore = 'MOVIE_SENTIMENT REQ_MORE'
recommend_reqmore = 'MOVIE_RECOMMEND REQ_MORE'

# Control Code for Goodbye
sentiment_goodbye = 'MOVIE_SENTIMENT GOODBYE'
recommend_goodbye = 'MOVIE_RECOMMEND GOODBYE'

# System Message for OOS Intent
oos_response_no_active_intent = "I'm sorry, I couldn't help you with your request at the moment. " \
                                "Do you want me to find you some movie recommendations or I can also help you to find " \
                                "out what people has been discussing about a movie."
oos_response_with_active_intent = "I'm sorry, I couldn't understand your request at the moment. Could you try again?"
