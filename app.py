import streamlit as st
from transformers import pipeline

# Initialize pipelines using CPU
sentiment_pipeline = pipeline("sentiment-analysis", device=-1)
summarizer = pipeline("summarization", device=-1)

st.title("🧠 Customer Review Analyzer")
st.markdown("Paste a review to analyze its sentiment and get a short summary.")

# Text input
review_text = st.text_area("✍️ Enter your review:", height=200)

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
