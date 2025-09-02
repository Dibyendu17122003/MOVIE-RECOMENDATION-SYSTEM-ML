import gradio as gr
import pickle
import pandas as pd
import requests
import time

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

# Custom CSS with modern design (adapted for Gradio)
custom_css = """
/* Main app styling */
.gradio-container {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%) !important;
    color: #ffffff !important;
    font-family: 'Poppins', sans-serif !important;
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
label {
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
}
.select-container {
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Button styling */
button {
    background: linear-gradient(135deg, #6a1b9a, #311d92) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 1rem !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 5px 15px rgba(106, 27, 154, 0.4) !important;
    position: relative !important;
    overflow: hidden !important;
}
button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(106, 27, 154, 0.6) !important;
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

/* Footer styling */
.footer {
    text-align: center;
    opacity: 0.7;
    font-size: 0.9rem;
    margin-top: 2rem;
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
"""

# Main function to handle recommendations
def get_recommendations(movie_name):
    if not movie_name:
        return [None] * 15  # Return empty values if no movie selected
    
    names, posters, links = recommend(movie_name)
    
    # Return all values in a flat list (5 names, 5 posters, 5 links)
    result = []
    for i in range(5):
        result.append(names[i])
        result.append(posters[i])
        result.append(links[i])
    
    return result

# Create the Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Default()) as demo:
    # App Header
    gr.HTML("""
        <div class="header">
            <h1>🎬 CineAI</h1>
            <p>Intelligent movie recommendations powered by AI</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            movie_dropdown = gr.Dropdown(
            choices=movies['title'].tolist(),  
            label="Select a movie you enjoy:",
            info="Choose a movie to discover similar recommendations"
        )
            
            # Recommendation button
            recommend_btn = gr.Button("Discover Recommendations")
    
    # Results section
    with gr.Column(visible=False) as results_col:
        gr.HTML("""
            <div class="recommendation-container">
                <h2 class="section-title">Recommended For You</h2>
            </div>
        """)
        
        # Create 5 columns for movie recommendations
        with gr.Row():
            rec_cols = []
            for i in range(5):
                with gr.Column(min_width=200) as col:
                    rec_image = gr.Image(label="", interactive=False, show_label=False, height=300)
                    rec_title = gr.HTML()
                    rec_link = gr.HTML()
                    rec_cols.extend([rec_title, rec_image, rec_link])
    
    # Footer
    gr.HTML("""
        <div class="footer">
            <p>Powered by Gradio • Movie data from OMDB • © 2025 • made by paradox</p>
        </div>
    """)
    
    # Event handling
    def show_recommendations(movie_name):
        if not movie_name:
            return {results_col: gr.update(visible=False)}
        
        results = get_recommendations(movie_name)
        
        # Update the recommendation columns
        updates = {}
        for i in range(5):
            # Update title
            updates[rec_cols[i*3]] = gr.update(value=f'<div class="movie-title">{results[i*3]}</div>')
            # Update image
            updates[rec_cols[i*3+1]] = gr.update(value=results[i*3+1])
            # Update link
            updates[rec_cols[i*3+2]] = gr.update(
                value=f'<a class="movie-link" href="{results[i*3+2]}" target="_blank">⭐ IMDb</a>'
            )
        
        # Show the results section
        updates[results_col] = gr.update(visible=True)
        
        return updates
    
    # Connect the button click event
    recommend_btn.click(
        fn=show_recommendations,
        inputs=[movie_dropdown],
        outputs=[rec_cols[0], rec_cols[1], rec_cols[2], 
                 rec_cols[3], rec_cols[4], rec_cols[5],
                 rec_cols[6], rec_cols[7], rec_cols[8],
                 rec_cols[9], rec_cols[10], rec_cols[11],
                 rec_cols[12], rec_cols[13], rec_cols[14],
                 results_col]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
