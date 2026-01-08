import streamlit as st

st.set_page_config(page_title="Word Counter", layout="centered")

st.title("📝 Simple Word Counter")

st.write("Type or paste some text below, and I will count how many words it has.")

text = st.text_area("Your text:", height=200)

# Simple word count: split on whitespace and ignore empty chunks
if st.button("Count words"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        words = [w for w in text.split() if w.strip()]
        word_count = len(words)

        st.success(f"Word count: **{word_count}**")