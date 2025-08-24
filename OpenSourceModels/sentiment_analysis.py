import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

# Use a pipeline as a high-level helper
from transformers import pipeline

analyzer = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def sentiment_analyzer(review):
    sentiment = analyzer(review)
    return sentiment[0]['label']

def sentiment_bar_chart(df):
    sentiment_counts = df['Sentiment'].value_counts()
    # Create a pie chart
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', color=['green', 'red'])
    ax.set_title('Review Sentiment Counts')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    return fig


def read_reviews_and_analyze_sentiment(file_object):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_object)
    # Apply the get_sentiment function to each review in the DataFrame
    df['Sentiment'] = df['Reviews'].apply(sentiment_analyzer)
    chart_object = sentiment_bar_chart(df)
    return df, chart_object

demo = gr.Interface(fn=read_reviews_and_analyze_sentiment,
                    inputs=[gr.File(label="Upload your review comment file")],
                    outputs=[gr.Dataframe(label="Sentiments"), gr.Plot(label="Sentiment Analysis")],
                    title="Sentiment Analyzer",
                    description="Sentiment Analyzer")
demo.launch()
