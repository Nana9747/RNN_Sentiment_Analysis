import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model 

## Load the word Index 
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}

## Load the pre-trained with Relu activation
model = load_model('simple_rnn_imdb_updated.h5')

# function to decode review
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

## Functions to preprocess user input
def preprocess_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word, 2) + 3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
  return padded_review

## Prediction Function 
def predict_sentiment(review):
  preprocessed_input = preprocess_text(review)
  #model.predict(preprocessed_input)

  prediction = model.predict(preprocessed_input)

  sentiment = 'positive' if prediction[0][0] > 0.5 else 'Negative'

  return sentiment, prediction[0][0]

##Design streamlit app 
 
import streamlit as st 
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

## User input 
user_input = st.text_area('Movie Review')

if st.button('Classify'):
  preprocess_input = preprocess_text(user_input)

  ## Make Prediction 
  prediction = model.predict(preprocess_input)
  sentiment = 'positive' if prediction[0][0] > 0.5 else 'Negative'

  ## Display Result 
  st.write(f'sentiment: {sentiment}')
  st.write(f'Prediction Score: {prediction[0][0]}')

else:
  st.write('Please enter a Movie review.')
    
