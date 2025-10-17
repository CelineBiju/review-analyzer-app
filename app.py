import streamlit as st
from transformers import pipeline

# Initialize pipelines with CPU
sentiment_pipeline = pipeline("sentiment-analysis", device=-1)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

# Streamlit app layout
st.set_page_config(page_title="Customer Review Analyzer", layout="centered")
st.title("🧠 Customer Review Analyzer")
st.markdown(
    "Enter a customer review below. The app will:\n"
    "- Analyze its **sentiment** (Positive/Negative)\n"
    "- Generate a **summary** (if the review is long enough)"
)

# User input
review_text = st.text_area("✍️ Paste your review here:", height=200)

# Button click
if st.button("Analyze Review") and review_text.strip():

    # Run sentiment analysis
    sentiment_result = sentiment_pipeline(review_text)[0]
    sentiment_label = sentiment_result['label']
    sentiment_score = sentiment_result['score']

    # Show sentiment result
    st.subheader("🔍 Sentiment Analysis")
    st.write(f"**Sentiment:** {sentiment_label} ({sentiment_score:.2f})")

    # Show original review
    st.subheader("📄 Original Review")
    st.write(review_text)

    # Summarization: only if review is long enough
    if len(review_text.split()) > 30:
        try:
            summary_result = summarizer(
                review_text,
                max_length=30,
                min_length=10,
                do_sample=False
            )[0]
            summary = summary_result['summary_text']
        except Exception as e:
            summary = "⚠️ Could not generate summary due to an error."
    else:
        summary = "✏️ Review too short for summarization. Here's the original review."

    # Show summary
    st.subheader("📝 Summary")
    st.write(summary)

else:
    st.info("Please enter a review and click **Analyze Review**.")
