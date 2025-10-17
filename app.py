import streamlit as st
from transformers import pipeline
import traceback

# Specify models explicitly
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1
    )
except Exception as e:
    st.error("Error loading models:")
    st.text(traceback.format_exc())
    st.stop()

st.title("🧠 Customer Review Analyzer")
st.markdown("Enter a customer review below. The app will analyze its sentiment, generate a summary, and guess the topic.")

review_text = st.text_area("✍️ Paste your review here:", height=200)

if st.button("Analyze Review") and review_text.strip():
    try:
        sentiment_result = sentiment_pipeline(review_text)[0]
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']

        summary_result = summarizer(review_text, max_length=50, min_length=25, do_sample=False)[0]
        summary = summary_result['summary_text']

        st.subheader("🔍 Analysis Results")
        st.write("**Sentiment:**", f"{sentiment_label} ({sentiment_score:.2f})")
        st.write("**Summary:**", summary)
    except Exception as e:
        st.error("Error during analysis:")
        st.text(traceback.format_exc())
else:
    st.info("Please enter a review and click 'Analyze Review'.")
