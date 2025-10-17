import os
# Disable file watchers to avoid watch limit error
os.environ['STREAMLIT_WATCHDOG_PATHS'] = "[]"

import streamlit as st
from transformers import pipeline

# Initialize pipelines (CPU only)
sentiment_pipeline = pipeline("sentiment-analysis", device=-1)
summarizer = pipeline("summarization", device=-1)

# Layout
st.set_page_config(page_title="Review Analyzer", layout="centered")
st.title("🧠 Customer Review Analyzer")

st.markdown("""
This app:
- ✅ Detects sentiment (Positive / Negative)
- ✏️ Summarizes reviews if they’re long enough
""")

review = st.text_area("✍️ Paste your review:", height=200)

if st.button("Analyze Review") and review.strip():
    # Sentiment
    try:
        sentiment = sentiment_pipeline(review)[0]
        st.subheader("💬 Sentiment")
        st.write(f"**Sentiment:** {sentiment['label']} ({sentiment['score']:.2f})")
    except Exception as e:
        st.error(f"Sentiment analysis error: {e}")

    # Summary
    st.subheader("📄 Summary")
    word_count = len(review.split())
    if word_count < 40:
        st.info("Review too short for summarization.")
    else:
        try:
            summary = summarizer(
                review,
                max_length=60,
                min_length=25,
                do_sample=False
            )[0]['summary_text']
            st.write(summary)
        except Exception as e:
            st.error(f"Summarization error: {e}")

else:
    st.info("Please paste a review and click 'Analyze Review'.")
