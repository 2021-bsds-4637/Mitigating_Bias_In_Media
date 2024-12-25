import streamlit as st
import joblib
import os

# Load the model and vectorizer
@st.cache_resource  # Cache the resources to avoid reloading them on every interaction
def load_resources():
    try:
        model = joblib.load('fake_news_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError as e:
        st.error("Required files not found. Ensure 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as this script.")
        raise e

model, vectorizer = load_resources()

# Title of the app
st.title("📰 Factify-Fake News Detector")


# Input box for the user to enter news text
user_input = st.text_area("Enter the News Article here:", "")

# Prediction function
def predict_news(text):
    """Predict whether the given news text is fake or real."""
    try:
        transformed_text = vectorizer.transform([text])  # Transform the input text
        prediction = model.predict(transformed_text)     # Make prediction
        return "Real News" if prediction[0] == 1 else "Fake News"
    except Exception as e:
        st.error("An error occurred during prediction. Please check the input or model.")
        raise e

# Button to make predictions
if st.button("Predict"):
    if user_input.strip():  # Ensure input is not empty
        prediction_result = predict_news(user_input)
        st.success(f"Prediction: **{prediction_result}**")
    else:
        st.warning("Please Enter some text to make joa prediction.")

# Footer
st.write("---")