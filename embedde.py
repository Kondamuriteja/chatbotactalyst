import openai
import pandas as pd
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embeddings(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Read the CSV file
news_df = pd.read_csv('articles.csv')

# Generate embeddings for each summary
news_df['embedding'] = news_df['summary'].apply(lambda x: get_embeddings(x) if pd.notnull(x) else None)

# Convert the list of embeddings to a JSON string
news_df['embedding'] = news_df['embedding'].apply(lambda x: json.dumps(x) if x is not None else None)

# Save the data with embeddings to a new CSV file
news_df.to_csv('news_data_with_embeddings.csv', index=False)
