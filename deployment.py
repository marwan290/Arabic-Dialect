
MODEL_PATH = r'models\mnb.pkl'
BOW_PATH = r'models\BOW.pkl'

import numpy as np
import re
import pickle
from flask import Flask,render_template,request


def normalize_arabic(text):
    # text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text
    
    
def remove_repeating_char(text):
    return re.sub(r'([^ل])\1+', r'\1', text)


def processPost(tweet): 

    #Replace @username with empty string
    tweet = re.sub('@[^\s]+', ' ', tweet)
    
    # remove Special Char
    tweet= re.sub(r'[`~!@#$%^&*()_|+\-=?؟،؛;:\'",.<>\{\}\[\]\\\/]', r' ', tweet)

    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    #remove English word and number
    tweet= re.sub(r'[0-9A-z]+', r' ', tweet)
    
    # normalize the tweet
    tweet= normalize_arabic(tweet)
    
    # remove repeated letters
    tweet=remove_repeating_char(tweet)

    #remove emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"٠١٢٣٤٥٦٧٨٩"   
        u"🤔🤣☺️✨☕️ ♥️👩‍❤️‍💋‍👩"          #remove arabic number
                           "]+", flags=re.UNICODE)  
    tweet= emoji_pattern.sub(r' ', tweet) # no emoji

    #remove extra space
    _RE_COMBINE_WHITESPACE = re.compile(r"(?a:\s+)")
    _RE_STRIP_WHITESPACE = re.compile(r"(?a:^\s+|\s+$)")

    tweet = _RE_COMBINE_WHITESPACE.sub(" ", tweet)
    tweet = _RE_STRIP_WHITESPACE.sub("", tweet)

    
    return tweet
    
    

model = pickle.load(open(MODEL_PATH, 'rb'))
bow = pickle.load(open(BOW_PATH, 'rb'))



app = Flask(__name__)

@app.route('/',methods=['GET'])
def hello():

    return render_template( 'index.html')

@app.route('/',methods=['POST'])
def predict():
    tweet = request.values['tweet']
    tweet = processPost(tweet)
    if tweet == "":
        return render_template( 'index.html' ,pred ="  ")
    
    tweet = bow.transform([tweet])
    return render_template( 'index.html' ,pred =model.predict(tweet)[0])

app.run(port=3000,debug=True)