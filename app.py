import streamlit as st
from transformers import pipeline

# Load models using CPU
sentiment_pipeline = pipeline("sentiment-analysis", device=-1)
summarizer = pipeline("summarization", device=-1)

# App layout
st.set_page_config(page_title="Review Analyzer", layout="centered")
st.title("🧠 Customer Review Analyzer")
st.markdown("""
Enter a customer review. This app will:
- ✅ Analyze the sentiment (Positive/Negative)
- ✏️ Summarize the review if it's long
""")

# Input
review = st.text_area("✍️ Paste your review here:", height=200)

if st.button("Analyze Review") and review.strip():
    # --- Sentiment Analysis ---
    sentiment = sentiment_pipeline(review)[0]
    st.subheader("💬 Sentiment Analysis")
    st.write(f"**Sentiment:** {sentiment['label']} ({sentiment['score']:.2f})")

    # --- Summarization ---
    st.subheader("📄 Summary")
    word_count = len(review.split())

    if word_count > 30:
        try:
            summary = summarizer(review, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
            st.write(summary)
        except Exception as e:
            st.warning("⚠️ Couldn't summarize this review.")
    else:
        st.info("Review too short for summarization.")
else:
    st.info("Please paste a review and click 'Analyze Review'.")
