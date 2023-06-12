import streamlit as st
import pandas as pd
import base64

import pickle
import nltk
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# functions to use on streamlit app
# import and set an image as background
# source: https://levelup.gitconnected.com/how-to-add-a-background-image-to-your-streamlit-app-96001e0377b2
def background(image_file):

    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# return a predicted book's details
# source: https://discuss.streamlit.io/t/image-and-text-next-to-each-other/7627
def book_details(predicted_book, df):
    found_book = []
    for book in predicted_book:
        book_detail = df[df['Title'] == book]

        if not book_detail.empty:
            book_title = book_detail['Title'].values[0]
            book_author = book_detail['authors'].values[0]
            book_description = book_detail['description'].values[0]
            book_cover = book_detail['image'].values[0]

            col1, col2 = st.columns([1,3])

            with col1:
                st.image(book_cover, caption = book, use_column_width = True)
            with col2:
                st.write(f'Title: {book_title}')
                st.write(f'Author(s): {book_author}')
                st.write(f'Description: {book_description}')

            found_book.append(book_title)
    if not found_book:
        st.write('Book details not found.')
    return found_book

# returns an author you should read
def author_books(predicted_author, df):
    found_books = []
    for author in predicted_author:
        book_details = df[df['authors'] == author]

        if not book_details.empty:
            book_title = book_details['Title'].values[0]
            book_author = book_details['authors'].values[0]
            book_description = book_details['description'].values[0]
            book_cover = book_details['image'].values[0]

            col1, col2 = st.columns([1, 3])

            with col1:
                st.image(book_cover, caption=book_title, use_column_width=True)
            with col2:
                st.write(f'Title: {book_title}')
                st.write(f'Author(s): {book_author}')
                st.write(f'Description: {book_description}')

            found_books.append(book_title)
    if not found_books:
        st.write('Book details not found.')
    return found_books

# lemmatizes inputted text to dictionary form to filder out and remove 
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


# predictions for Fiction genre, returns a book title & description
def fiction_pred(fic_fp, data):
    with open(fic_fp, 'rb') as pickle_in:
        fic_pipe = pickle.load(pickle_in)

    text = st.text_input('Please enter a description of a possible book (Fiction): ', max_chars = 500)

    if text:
        predicted_book = fic_pipe.predict([text])[0]
        st.write(f'You should read {predicted_book.title()}.')
        return book_details([predicted_book], data)

# predictions for Juvenile Fiction, returns a book title & descriptions
def jvf_pred(jvf_fp, data):
    with open(jvf_fp, 'rb') as pickle_in:
        jvf_pipe = pickle.load(pickle_in)

    text = st.text_input('Please enter a description of a possible book (Juvenile Fiction): ', max_chars = 500)

    if text:
        predicted_book = jvf_pipe.predict([text])[0]
        st.write(f'You should read {predicted_book.title()}.')
        return book_details([predicted_book], data)

# predictions for Biography & Autobiography genre, returns a book title & description
def bio_pred(bio_fp, data):
    with open(bio_fp, 'rb') as pickle_in:
        bio_pipe = pickle.load(pickle_in)

    text = st.text_input('Please enter a description of a possible book (Biography & Autobiography): ', max_chars = 500)

    if text:
        predicted_book = bio_pipe.predict([text])[0]
        st.write(f'You should read {predicted_book.title()}.')
        return book_details([predicted_book], data)

# predictions for most reviewed books
def review_pred(review_fp, data):
    with open(review_fp, 'rb') as pickle_in:
        review_pipe = pickle.load(pickle_in)

    text = st.text_input('Please enter a description of a possible book: ', max_chars = 500)

    if text:
        predicted_author = review_pipe.predict([text])[0]
        st.write(f'You should read {predicted_author.title()}.')
        return author_books([predicted_author], data)

# predictions for overall dataset
# I will use this for my random mode since the model was really bad
def overall_pred(overall_fp, data):
    with open(overall_fp, 'rb') as pickle_in:
        overall_pipe = pickle.load(pickle_in)

    text = st.text_input('Please enter a description of a possible book: ', max_chars = 500)

    if text:
        predicted_author = overall_pipe.predict([text])[0]
        st.write(f'You should read {predicted_author.title()}.')
        return author_books([predicted_author], data)
