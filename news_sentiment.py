from newsapi import NewsApiClient
from transformers import pipeline
import re

def fetch_news_sentiment(ticker):
    newsapi = NewsApiClient(api_key='56123a51b7234785a902fc868cff9d0b')
    articles = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy')
    
    sentiment_analyzer = pipeline('sentiment-analysis')
    sentiments = []
    for article in articles['articles']:
        if re.search(r'\b(business|market|finance|economy)\b', article['title'], re.IGNORECASE):
            text = article['title'] + " " + article['description']
            sentiment = sentiment_analyzer(text)[0]
            score = sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
            sentiments.append(score)
    
    average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    return average_sentiment
