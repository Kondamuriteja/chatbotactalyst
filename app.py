import streamlit as st
import pandas as pd
import json
from querryhandling import find_most_relevant_article

# Load the data with embeddings
news_df = pd.read_csv('news_data_with_embeddings.csv')
news_df['embedding'] = news_df['embedding'].apply(lambda x: json.loads(x) if pd.notnull(x) else None)

st.title("Aluminium Industry News Chatbot")
st.write("Ask me anything about the latest news in the aluminium industry!")

user_query = st.text_input("Enter your query:")

if user_query:
    with st.spinner("Searching for relevant articles..."):
        most_relevant_article = find_most_relevant_article(user_query, news_df)
        st.write("### Most Relevant Article")
        st.write(f"**Title:** {most_relevant_article['title']}")
        st.write(f"**Summary:** {most_relevant_article['summary']}")
        st.write(f"**Publication Date:** {most_relevant_article['date']}")
        
