import os
import requests
import streamlit as st
import ai21
from dotenv import load_dotenv
import datetime

# Load secrets
load_dotenv()

# Set up ai21 API key 
# ai21.api_key = os.getenv("AI21_API_KEY")
ai21.api_key = '5UmK80DxqWV1sXhsLO66zBrNSrMaOvFA' # 'your_ai21_api_key_here'

# Set up News API endpoint
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
url = f"https://newsapi.org/v2/everything?apiKey={NEWS_API_KEY}"

st.title("NewsTrackr: AI News Aspect Based Sentiment Analysis")

# Get user topic input
topic = st.text_input("Enter the NEWS title for aspect-based sentiment analysis")
start = st.text_input("Please specify the date from which you would like to collect the news articles. (YYYY-MM-DD)")
end = st.text_input("Please specify the date till which you would like to collect the news articles. (YYYY-MM-DD)")

if st.button("Search"):
    # Set query parameters and fetch news articles from the API
    params = {"q": topic, "sortBy": "relevancy", "language": "en", "from": start, "to": end} # 
    response = requests.get(url, params=params)
    articles = response.json()["articles"]

    # Process the articles and get predicted sports from AI model
    for article in articles:
        title = article["title"]
        content = article["content"]
        url = article['url']
        
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
        st.write(f"Article URL: {url}")
        st.write("---")


        
