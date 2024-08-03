import openai
import pandas as pd
import json
from dotenv import load_dotenv
import os
import requests

# Load environment variables from .env file
load_dotenv()

# Retrieve values from environment variables
api_endpoint = os.getenv('OPENAI_KEY_API_ENDPOINT')
email = os.getenv('EMAIL')

# Function to get OpenAI keys
def get_openai_keys(api_endpoint, email):
    payload = {"email": email}
    headers = {"Content-Type": "application/json"}
    response = requests.post(api_endpoint, data=json.dumps(payload), headers=headers)
    if response.status_code == 200:
        keys = response.json()
        return keys['key']
    else:
        raise Exception(f"Failed to retrieve keys: {response.text}")

# Retrieve OpenAI API key
key = get_openai_keys(api_endpoint, email)

# Set OpenAI API key
openai.api_key = key

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

# Convert the list of embeddings to a format that can be saved in a CSV
news_df['embedding'] = news_df['embedding'].apply(lambda x: json.dumps(x) if x is not None else None)

# Save the data with embeddings to a new CSV file
news_df.to_csv('news_data_with_embeddings.csv', index=False)
