import streamlit as st
from transformers import pipeline

# Initialize pipelines to use CPU only (device=-1)
sentiment_pipeline = pipeline("sentiment-analysis", device=-1)
summarizer = pipeline("summarization", device=-1)

st.title("🧠 Customer Review Analyzer")

st.markdown("Enter a customer review below. The app will analyze its sentiment, generate a summary, and guess the topic.")

review_text = st.text_area("✍️ Paste your review here:", height=200)

if st.button("Analyze Review") and review_text.strip():
    sentiment_result = sentiment_pipeline(review_text)[0]
    sentiment_label = sentiment_result['label']
    sentiment_score = sentiment_result['score']

    try:
        summary_result = summarizer(review_text, max_length=50, min_length=25, do_sample=False)[0]
        summary = summary_result['summary_text']
    except Exception:
        summary = "Could not generate summary. Review might be too short or too long."

    st.subheader("🔍 Analysis Results")
    st.write("**Sentiment:**", f"{sentiment_label} ({sentiment_score:.2f})")
    st.write("**Summary:**", summary)
else:
    st.info("Please enter a review and click 'Analyze Review'.")
