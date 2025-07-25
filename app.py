import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

# Load the saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("📰 Fake News Detection App")

# Input text from user
user_input = st.text_area("Enter the news article text:")

# Button
if st.button("Check if it's Fake or Real"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)

        # Output result
        if prediction[0] == 0:
            st.success("✅ This news article is REAL.")
        else:
            st.error("🚨 This news article is FAKE.")
