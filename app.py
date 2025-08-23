import streamlit as st
import pickle
import pandas as pd
import requests
import time
from streamlit.components.v1 import html

# OMDb API key
OMDB_KEY = "a1603e57"

# Function to fetch movie poster + IMDb link
def fetch_poster(movie_title):
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={OMDB_KEY}"
    response = requests.get(url).json()
    poster_url = response.get("Poster")
    imdb_id = response.get("imdbID")
    
    if poster_url and poster_url != "N/A":
        poster = poster_url
    else:
        poster = "https://images.unsplash.com/photo-1542204165-65bf26472b9b?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=80"

    imdb_link = f"https://www.imdb.com/title/{imdb_id}" if imdb_id else "#"
    return poster, imdb_link

# Function to recommend movies
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6] 
    
    recommended_movies = []
    recommended_movies_poster = []
    recommended_movies_links = []
    for i in movies_list:
        movie_title = movies.iloc[i[0]].title
        poster, link = fetch_poster(movie_title)
        recommended_movies.append(movie_title)
        recommended_movies_poster.append(poster)
        recommended_movies_links.append(link)

    return recommended_movies, recommended_movies_poster, recommended_movies_links

# Load data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Custom CSS with modern design
st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Header styling */
    .header {
        text-align: center;
        padding: 2.5rem 0;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, 
            rgba(106, 27, 154, 0.3) 0%, 
            rgba(49, 27, 146, 0.3) 50%, 
            rgba(26, 35, 126, 0.3) 100%);
        z-index: -1;
        border-radius: 0 0 20px 20px;
    }
    .header h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #6a1b9a, #311d92, #1a237e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    .header p {
        font-size: 1.2rem;
        opacity: 0.8;
        margin: 0.5rem 0 0 0;
    }
    
    /* Select box styling */
    .stSelectbox label {
        color: white !important;
        font-size: 1.1rem;
        font-weight: 500;
    }
    div[data-baseweb="select"] {
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #6a1b9a, #311d92);
        color: white;
        font-weight: 600;
        border: none;
        padding: 1rem;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(106, 27, 154, 0.4);
        position: relative;
        overflow: hidden;
    }
    .stButton button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.2), 
            transparent);
        transition: all 0.5s ease;
    }
    .stButton button:hover::before {
        left: 100%;
    }
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(106, 27, 154, 0.6);
    }
    
    /* Movie card styling */
    .movie-card {
        text-align: center;
        padding: 1.2rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #6a1b9a, #311d92, #1a237e);
    }
    .movie-card:hover {
        transform: translateY(-8px) scale(1.02);
        background: rgba(255, 255, 255, 0.08);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    .movie-poster {
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .movie-card:hover .movie-poster {
        transform: scale(1.05);
    }
    .movie-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #fff;
        margin: 1rem 0;
        line-height: 1.4;
        height: 3rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .movie-link {
        display: inline-block;
        background: linear-gradient(135deg, #6a1b9a, #311d92);
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 25px;
        text-decoration: none;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.3s ease;
        margin-top: auto;
        box-shadow: 0 4px 10px rgba(106, 27, 154, 0.3);
    }
    .movie-link:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 15px rgba(106, 27, 154, 0.4);
    }
    
    /* Section styling */
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1.5rem 0;
        background: linear-gradient(135deg, #6a1b9a, #311d92);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Animation for recommendations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .recommendation-container {
        animation: fadeIn 0.8s ease forwards;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .header h1 {
            font-size: 2.5rem;
        }
        .movie-card {
            margin-bottom: 1.5rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6a1b9a, #311d92);
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# JavaScript for animations
st.markdown("""
    <script>
    // Simple typing effect for the header
    document.addEventListener('DOMContentLoaded', function() {
        const titleElement = document.querySelector('.header h1');
        const originalText = titleElement.textContent;
        titleElement.textContent = '';
        
        let i = 0;
        function typeWriter() {
            if (i < originalText.length) {
                titleElement.textContent += originalText.charAt(i);
                i++;
                setTimeout(typeWriter, 100);
            }
        }
        typeWriter();
    });
    </script>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
    <div class="header">
        <h1>🎬 CineAI</h1>
        <p>Intelligent movie recommendations powered by AI</p>
    </div>
""", unsafe_allow_html=True)

# Movie selection with custom styling
selected_movie_name = st.selectbox(
    'Select a movie you enjoy:',
    movies['title'].values,
    help="Choose a movie to discover similar recommendations"
)

# Recommendation button
if st.button('Discover Recommendations', key="recommend_btn"):
    with st.spinner('Analyzing thousands of movies to find your perfect matches...'):
        # Simulate processing time for a more dramatic effect
        time.sleep(1.5)
        names, posters, links = recommend(selected_movie_name)
    
    # Recommendations section
    st.markdown("""
        <div class="recommendation-container">
            <h2 class="section-title">Recommended For You</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Create columns for movie cards
    cols = st.columns(5)
    
    for idx, col in enumerate(cols):
        with col:
            st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-poster">
                        <img src="{posters[idx]}" width="100%" style="display:block;"/>
                    </div>
                    <div class="movie-title">{names[idx]}</div>
                    <a class="movie-link" href="{links[idx]}" target="_blank">⭐ IMDb</a>
                </div>
            """, unsafe_allow_html=True)

# Add footer
st.markdown("""
    <br><br>
    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
        <p>Powered by Streamlit • Movie data from OMDB • © 2025 • made by paradox</p>
    </div>
""", unsafe_allow_html=True)