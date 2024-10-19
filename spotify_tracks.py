import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('SpotifyMostStreamed.csv')

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Define query songs based on their titles
query_songs = ['Shape of You', 'Blinding Lights', 'Dance Monkey']
query_indices = df[df['Artist and Title'].str.contains('|'.join(query_songs))].index.tolist()

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the song titles to TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(df['Artist and Title'])

# Compute cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix)

# Function to get top N similar songs for each query
def get_top_similar_songs(query_index, n=10):
    # Get similarity scores for the query song
    similarity_scores = similarity_matrix[query_index]
    
    # Get indices of the top N similar songs, excluding the song itself
    similar_indices = similarity_scores.argsort()[-n-1:-1][::-1]
    
    # Create a DataFrame for easier handling
    similar_songs = df.iloc[similar_indices].copy()
    similar_songs['Similarity Score'] = similarity_scores[similar_indices]
    
    return similar_songs

# Get top 10 similar songs for each query and print them
for index in query_indices:
    top_similar_songs = get_top_similar_songs(index)
    print(f"\nTop 10 songs similar to '{df.iloc[index]['Artist and Title']}':\n")
    print(top_similar_songs[['Artist and Title', 'Streams', 'Daily', 'Similarity Score']])