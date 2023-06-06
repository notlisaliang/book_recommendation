import streamlit as st
import base64
import pickle
import pandas as pd

st.set_page_config(
    page_title = 'Book Recommendation',
    page_icon = '📚'
)

st.title('Homepage')

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
background('coverpage.png')  

def fiction_pred():
    fiction_fp = "/Users/lisaliang/my-capstone/capstone/streamlit_app/fiction_pipe.pkl"

    with open(fiction_fp, 'rb') as pickle_in:
        fiction_model = pickle.load(pickle_in)

    user_text = st.text_input("Please enter some text:", max_chars = 500)

    if user_text: 
        predicted_book = model.predict([user_text])
        st.write(f'You should read {predicted_book}')

# source: https://stackoverflow.com/questions/69492406/streamlit-how-to-display-buttons-in-a-single-line
genre_col, overall_col, chaos_col = st.columns([1, 1, 1])

with genre_col:
    if st.button('Genre'):
        st.selectbox('Select a genre: ', ("Fiction", "Juvenile Fiction", "Biography & Autobiography"))
with overall_col:
    st.button('Overall')
with chaos_col:
    st.button('Chaos')



#st.title('Genre')

# idea: pick the genre and it'll direct to the path/model to do the suggestion

#file_path = "/Users/lisaliang/Documents/projects/capstone/streamlit_app/fiction_pipe.pkl"

#with open(file_path, 'rb') as pickle_in:
    #model = pickle.load(pickle_in)

#df = pd.read_csv('fiction_sample.csv')

#user_text = st.text_input("Please enter some text:", max_chars = 500)

#predicted_books = pipe.predict([your_text])[:10]

#if user_text: 
    #predicted_book = model.predict([user_text])
    #st.write(f'You should read {predicted_book}')

#for predicted_book in predicted_books:
    #book_row = df[df['Title'] == predicted_book]

    #if not book_row.empty:
        #book_cover_url = book_row['image'].values[0]
        #st.image(book_cover_url, caption=predicted_book, use_column_width=True)
    #else:
        #st.write("Book cover not found.")

    #st.write(f'You should read {predicted_book}')
    #st.write('---')



