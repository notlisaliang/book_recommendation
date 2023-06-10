# imports
import streamlit as st
import pandas as pd
import base64

import pickle
import nltk
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import functions

st.set_page_config(
    page_title = 'Book Recommendation',
    page_icon = '📚'
)

st.sidebar.success('Welcome!')

st.title('What book should I read?')

# setting up background
functions.background('coverpage.png') 

st.write('Welcome to Your Little Corner! Are you looking for your next favorite book?')
st.write('Go to the sidebar to view your options. You can receive recommendations by genre or through highly reviewed books.')
st.write('Enjoy your stay!')

# unsure why, but I need this on the same py file I call the other functions for lemmatization to work
def my_lemmatizer(text):
    wnet = WordNetLemmatizer()
    # exclude words with apostrophes and numbers
    return [wnet.lemmatize(w) for w in text.split() if "'" not in w and not w.isdigit()]

# setting up stopwords for pickled models
wnet = WordNetLemmatizer()
lem_stopwords = [wnet.lemmatize(w) for w in stopwords.words('english')]

contractions = ['ve', 't', "'s'", 'd', 'll', 'm', 're']
lem_contractions = [wnet.lemmatize(contraction) for contraction in contractions]

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lem_numbers = [wnet.lemmatize(num) for num in numbers]

lem_stopwords = lem_stopwords + lem_contractions + lem_numbers


# read genre datasets
#fic_data = pd.read_csv("/Users/lisaliang/my-capstone/book_recommendation/data/fiction_sample.csv")
#jvf_data = pd.read_csv("/Users/lisaliang/my-capstone/book_recommendation/data/jvf_sample.csv")
#bio_data = pd.read_csv("/Users/lisaliang/my-capstone/book_recommendation/data/bio_sample.csv")

# create filepaths for pickled models
#fic_fp = "/Users/lisaliang/my-capstone/book_recommendation/streamlit_app/pickled_models/fiction_pipe.pkl"
#jvf_fp = "/Users/lisaliang/my-capstone/book_recommendation/streamlit_app/pickled_models/jvf_pipe.pkl"
#bio_fp = "/Users/lisaliang/my-capstone/book_recommendation/streamlit_app/pickled_models/bio_pipe.pkl"