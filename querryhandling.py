import openai
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_query_embedding(query):
    response = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def find_most_relevant_article(query, df):
    query_embedding = get_query_embedding(query)
    embeddings = df['embedding'].apply(json.loads).tostring()
    similarities = cosine_similarity([query_embedding], embeddings).flatten()
    most_similar_index = np.argmax(similarities)
    most_relevant_article = df.iloc[most_similar_index]
    return most_relevant_article
