import streamlit as st
from transformers import pipeline
import traceback

st.set_page_config(page_title="Customer Review Analyzer", page_icon="📝")

st.title("📝 Customer Review Analyzer")
st.markdown("Analyze customer feedback using sentiment analysis and summarization.")

# User Input
review = st.text_area("Enter your customer review here:")

# Load models safely
@st.cache_resource
def load_models():
    try:
        sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # CPU
        )
        summary = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=-1  # CPU
        )
        return sentiment, summary
    except Exception as e:
        st.error("❌ Failed to load models.")
        st.text(traceback.format_exc())
        return None, None

sentiment_pipeline, summarizer = load_models()

if st.button("Analyze Review"):
    if not review.strip():
        st.warning("⚠️ Please enter a review before clicking 'Analyze Review'.")
    elif not sentiment_pipeline or not summarizer:
        st.error("❌ Models not loaded. Check logs.")
    else:
        try:
            # Sentiment
            sentiment_result = sentiment_pipeline(review)[0]
            label = sentiment_result['label']
            score = sentiment_result['score']

            st.subheader("🔍 Sentiment Analysis")
            st.write(f"**Sentiment:** {label} ({score:.2f})")

            # Summarization
            st.subheader("📄 Summary")
            if len(review.split()) < 30:
                st.info("ℹ️ Review too short to summarize.")
            else:
                summary_result = summarizer(
                    review,
                    max_length=60,
                    min_length=25,
                    do_sample=False
                )
                summary_text = summary_result[0]['summary_text']
                st.write(summary_text)

        except Exception as e:
            st.error("❌ An error occurred while processing the review.")
            st.text(traceback.format_exc())
