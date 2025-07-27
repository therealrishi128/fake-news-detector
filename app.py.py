import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Streamlit UI
st.title("📰 Fake News Detector")

# Text input from user
user_input = st.text_area("Enter the news content to check:")

# Predict on click
if st.button("Check if it's Fake or Real"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        transformed_input = vectorizer.transform([user_input])
        result = model.predict(transformed_input)

        if result[0] == 0:
            st.success("✅ Real News")
        else:
            st.error("🚨 Fake News")
