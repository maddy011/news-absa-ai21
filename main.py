import os
import requests
import streamlit as st
import ai21
from dotenv import load_dotenv

# Load secrets
load_dotenv()

API_KEY = os.getenv("AI21_LABS_API_KEY")

# Set up ai21 API key
ai21.api_key = 'NKz8j9xx3PpYS32lr40sPSm4oQJacbx0' # API_KEY

# Set up News API endpoint
NEWS_API_KEY = "b95dbe8383ec43739e32c71fbc516bb6"
url = f"https://newsapi.org/v2/everything?apiKey={NEWS_API_KEY}"

st.title("AI Sports News")

# Get user topic input
topic = st.text_input("Enter some index or ")

if st.button("Search"):
    # Set query parameters and fetch news articles from the API
    params = {"q": topic, "sortBy": "relevancy", "language": "en"}
    response = requests.get(url, params=params)
    articles = response.json()["articles"]

    # Process the articles and get predicted sports from AI model
    for article in articles:
        title = article["title"]
        content = article["content"]
        prompt = article["title"]
        response = ai21.Completion.execute(
            model="j2-large",
            custom_model="ASBA-j2-large-v2",
            prompt="find aspect based sentiment analysis for this text" + prompt,
            numResults=1,
            maxTokens=200,
            temperature=0.7,
            topKReturn=0,
            topP=1,
            countPenalty={
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False
                },
            frequencyPenalty={
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False
                },
            presencePenalty={
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False
                },
            stopSequences=[]
        )
        ABSA = response.completions[0].data.text          

        # Display the predicted sport for each article
        st.write(f"Article title: {title}")
        st.write(f"Aspect Sentiment: {ABSA}")
        st.write(f"Article content: {content}")
        st.write("---")


        
