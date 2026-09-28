import streamlit as st
import re

from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


# Load the trained model
model = load_model("sentiment_model.keras")


# Load the tokenizer
with open("tokenizer.json", "r") as file:
    tokenizer = tokenizer_from_json(file.read())


# Clean the movie review
def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# App title
st.title("🎬 Movie Review Sentiment Analyzer")

st.write(
    "Enter a movie review and the model will predict whether "
    "the sentiment is positive or negative."
)


# Review input
review = st.text_area(
    "Movie Review",
    placeholder="Example: I really enjoyed this movie!"
)


# Analyze button
if st.button("Analyze Sentiment"):

    if not review.strip():
        st.warning("Please enter a movie review.")

    else:
        # Clean the review
        cleaned = clean_text(review)

        # Convert words to numbers
        sequence = tokenizer.texts_to_sequences([cleaned])

        # Pad the sequence to 250 tokens
        padded = pad_sequences(
            sequence,
            maxlen=250,
            padding="post",
            truncating="post"
        )

        # Get prediction
        prediction = model.predict(padded, verbose=0)

        # Get positive probability
        probability = float(prediction[0][0])

        # Determine sentiment
        if probability >= 0.5:
            sentiment = "Positive"
            confidence = probability
        else:
            sentiment = "Negative"
            confidence = 1 - probability

        # Display result
        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2%}")