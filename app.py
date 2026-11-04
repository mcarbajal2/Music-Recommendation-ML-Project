import streamlit as st
import pandas as pd
import numpy as np
import os
#os.environ['OMP_NUM_THREADS'] = '1'
import requests
import spotipy
import lyricsgenius
import faiss
import streamlit.components.v1 as components
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from spotipy.oauth2 import SpotifyClientCredentials
from sentence_transformers import SentenceTransformer

#Page Setup
st.set_page_config(page_title="Music Recommendation ML P›roject", layout="wide")

#Environment & API Clients
load_dotenv()
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
))

genius = lyricsgenius.Genius(os.getenv("GENIUS_ACCESS_TOKEN"), timeout=15, retries=3)

#Resource Caching
@st.cache_resource
def load_models():
    model_instance = SentenceTransformer("BAAI/bge-m3")
    faiss_index = faiss.read_index("data/processed/lyric_embeddings.index")
    return model_instance, faiss_index

@st.cache_data
def load_dataframe():
    return pd.read_pickle("data/processed/final_merged_data.pkl")

@st.cache_data
def fetch_every_noise_genres():
    response = requests.get("https://everynoise.com/")
    soup = BeautifulSoup(response.text, "html.parser")
    genres = {div.text.strip().lower().replace('»', '').strip() for div in soup.find_all('div', class_='genre')}
    normalized = {g.replace('-', '').replace(' ', '').replace('_', '').replace('/', '').replace("'", ""): g for g in genres}
    custom = {
        'heavymetal': 'heavy metal', 'punkrock': 'punk rock', 'hardrock': 'hard rock',
        'britishpunk': 'british punk', 'rocknroll': 'rock and roll', 'rnb': 'r&b'
    }
    normalized.update(custom)
    return normalized

model, index = load_models()
final_df = load_dataframe()
normalized_noise_genres = fetch_every_noise_genres()

#Backend Logic
def split_genres_normalized(tag_list):
    if not isinstance(tag_list, list):
        return [], []
    genres, descriptors = [], []
    for tag in tag_list:
        lower_tag = tag.lower()
        norm_tag = lower_tag.replace('-', '').replace(' ', '').replace('_', '').replace('/', '').replace("'", "")
        if norm_tag in normalized_noise_genres:
            genres.append(normalized_noise_genres[norm_tag]) 
        else:
            descriptors.append(lower_tag)
    return list(set(genres)), descriptors

def fetch_lastfm_tags(artist, track, api_key):
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {'method': 'track.gettoptags', 'artist': artist, 'track': track, 'api_key': api_key, 'format': 'json'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        tags = [tag['name'] for tag in response.json().get('toptags', {}).get('tag', [])]
        if tags: return tags
    params['method'] = 'artist.gettoptags'
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return [tag['name'] for tag in response.json().get('toptags', {}).get('tag', [])]
    return []

def get_track_vector_and_meta(artist_name, track_title):    
    local_match = final_df[
        (final_df['artist'].str.lower() == artist_name.lower()) & 
        (final_df['song'].str.lower() == track_title.lower())
    ]
    if not local_match.empty:
        row_idx = local_match.index[0]
        return index.reconstruct(int(row_idx)), local_match.iloc[0]['genres']
    
    raw_tags = fetch_lastfm_tags(artist_name, track_title, LASTFM_API_KEY)
    clean_genres, _ = split_genres_normalized(raw_tags) 
    
    song = genius.search_song(track_title, artist_name)
    if not song or not song.lyrics:
        return None, None
        
    return model.encode(song.lyrics), clean_genres

def fetch_lastfm_user_profile(username, api_key, limit=20):
    url = "https://ws.audioscrobbler.com/2.0/"
    params = {'method': 'user.gettoptracks', 'user': username, 'api_key': api_key, 'format': 'json', 'limit': limit}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []
    return [{'artist': item['artist']['name'], 'song': item['name']} for item in response.json().get('toptracks', {}).get('track', [])]

def build_user_centroid(user_tracks):
    vectors = []
    for track in user_tracks:
        vector, _ = get_track_vector_and_meta(track['artist'], track['song'])
        if vector is not None:
            vectors.append(vector)
    if not vectors:
        raise ValueError("Could not resolve any vectors for the user profile.")
    return np.mean(np.array(vectors), axis=0).astype('float32')

def generate_recommendations(centroid, user_tracks, top_k=5, mode="similar"):
    search_depth = 50 if mode == "similar" else 150 
    distances, indices = index.search(np.array([centroid]), search_depth)
    recommendations = []
    user_artists = {track['artist'].lower() for track in user_tracks}
    
    for i, idx in enumerate(indices[0]):
        row = final_df.iloc[idx]
        artist = row['artist']
        if artist.lower() in user_artists:
            continue
        if mode == "discovery" and i < 20:
            continue
            
        recommendations.append({
            'artist': artist, 'song': row['song'], 'genres': row['genres'],
            'similarity_score': round(float(distances[0][i]), 4)
        })
        if len(recommendations) >= top_k:
            break
    return pd.DataFrame(recommendations)

@st.cache_data 
def fetch_spotify_assets(artist, track):
    query = f"artist:{artist} track:{track}"
    results = sp.search(q=query, type="track", limit=1)
    if not results['tracks']['items']:
        return None, None, None
    track_data = results['tracks']['items'][0]
    album_art = track_data['album']['images'][0]['url'] if track_data.get('album', {}).get('images') else None
    preview_url = track_data.get('preview_url')
    spotify_url = track_data.get('external_urls', {}).get('spotify')
    return album_art, preview_url, spotify_url

#Streamlit UI
st.title("🎵 Music Recommendation ML Project")

st.sidebar.header("User Profile")
username_input = st.sidebar.text_input("Last.fm Username", "")
generate_btn = st.sidebar.button("Generate Recommendations")

if generate_btn:
    with st.spinner("Analyzing profile and computing lyric embeddings..."):
        my_tracks = fetch_lastfm_user_profile(username_input, LASTFM_API_KEY)
        if my_tracks:
            centroid = build_user_centroid(my_tracks)
            similar_df = generate_recommendations(centroid, my_tracks, top_k=5, mode="similar")
            discovery_df = generate_recommendations(centroid, my_tracks, top_k=5, mode="discovery")
            
            tab1, tab2 = st.tabs(["🎧 Similar Tracks", "🚀 Discovery Tracks"])
            
            def render_track_cards(df):
                for _, row in df.iterrows():
                    art_url, preview, spot_link = fetch_spotify_assets(row['artist'], row['song'])
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if art_url:
                            st.image(art_url, width=300)
                        else:
                            st.write("💿 No Art Available")
                    with col2:
                        st.subheader(f"[{row['song']}]({spot_link})")
                        st.write(f"**Artist:** {row['artist']}")
                        st.write(f"**Genres:** {', '.join(row['genres']) if row['genres'] else 'Unknown'}")
                        if spot_link:
                            track_id = spot_link.split('/')[-1]
                            embed_url = f"https://open.spotify.com/embed/track/{track_id}"
                            components.html(
                            f'<iframe src="{embed_url}" width="100%" height="175" frameBorder="0" allow="encrypted-media"></iframe>',
                            height=175,
                        )
                        else:
                            st.caption("Spotify player unavailable for this track.")
                    st.divider()

            with tab1:
                render_track_cards(similar_df)
            with tab2:
                render_track_cards(discovery_df)
        else:
            st.error("Could not fetch tracks for this Last.fm user. Please check the username.")