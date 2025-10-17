import streamlit as st
from transformers import pipeline

# Load models (CPU only)
sentiment_pipeline = pipeline("sentiment-analysis", device=-1)
summarizer = pipeline("summarization", device=-1)

# Streamlit UI
st.set_page_config(page_title="Review Analyzer", layout="centered")
st.title("🧠 Customer Review Analyzer")

st.markdown("""
This app:
- ✅ Detects **sentiment**
- ✏️ Summarizes **long reviews** (over 40 words)
""")

# Input
review = st.text_area("✍️ Paste your review here:", height=200)

if st.button("Analyze Review") and review.strip():
    # --- Sentiment Analysis ---
    try:
        sentiment = sentiment_pipeline(review)[0]
        st.subheader("💬 Sentiment")
        st.write(f"**Sentiment:** {sentiment['label']} ({sentiment['score']:.2f})")
    except Exception as e:
        st.error(f"❌ Sentiment analysis failed: {e}")

    # --- Summarization ---
    st.subheader("📄 Summary")
    word_count = len(review.split())

    if word_count < 40:
        st.info("Review too short for summarization (need > 40 words).")
    else:
        try:
            summary = summarizer(
                review,
                max_length=60,
                min_length=25,
                do_sample=False
            )[0]['summary_text']
            st.success("✅ Summary generated:")
            st.write(summary)
        except Exception as e:
            st.error(f"⚠️ Summarization failed: {e}")
else:
    st.info("Please paste a review and click 'Analyze Review'.")
