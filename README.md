<div align="center">

# 🎬 Movie Recommendation System
**Content-Based Movie Recommender with End-to-End ML Pipeline**  

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-green?logo=streamlit)](https://streamlit.io/) 
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikit-learn)](https://scikit-learn.org/) 
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)  

</div>

---

## ✨ Project Demo

<p align="center">
<img src="assets/demo.gif" alt="Movie Recommendation Demo" width="700">
</p>

---

## 🚀 Features

<div align="center">

| Feature | Description |
|---------|-------------|
| 🎯 **Content-Based Filtering** | Cosine similarity recommendations |
| ⚡ **Hyperparameter Tuning** | GridSearchCV for TF-IDF vectorizer |
| 🔍 **Advanced Preprocessing** | Stemming, stopwords removal, feature engineering |
| 💻 **Interactive UI** | Streamlit frontend for easy interaction |
| ☁️ **Deployment Ready** | Ready for Render or other cloud platforms |
| 💾 **Model Persistence** | Save/load similarity matrices |

</div>

---

## 🛠️ Tech Stack

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)] 
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-yellow?logo=scikit-learn)]
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-lightblue?logo=pandas)]
[![NLTK](https://img.shields.io/badge/NLTK-NLP-orange?logo=nltk)]
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-green?logo=streamlit)]
[![Render](https://img.shields.io/badge/Deployment-Render-purple)]

</div>

---

## 📂 Project Structure

MOVIE-RECOMMENDATION-SYSTEM/
├── .gitignore
├── README.md
├── app.py # Streamlit frontend
├── movie-recommendation-system.ipynb # Jupyter Notebook
├── procfile # Deployment config
├── requirements.txt # Python dependencies
├── setup.sh # Setup script
├── tmdb_5000_credits.csv # Credits dataset
└── tmdb_5000_movies.csv # Movies dataset


---

## ⚡ Installation

<details>
<summary>Click to Expand Installation Steps</summary>

'bash'
# Clone repository
git clone https://github.com/yourusername/movie-recommendation-system.git
cd MOVIE-RECOMMENDATION-SYSTEM

# Run setup script (installs dependencies and downloads NLTK data)
bash setup.sh

Or manually:

# Create virtual environment
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download NLTK stopwords
python -c "import nltk; nltk.download('stopwords')"
</details>

📝 Data Preprocessing
<details> <summary>Click to Expand Preprocessing Steps</summary>
Pipeline includes:
✅ Cleaning: missing values, duplicates
📝 Text Processing: lowercase, remove stopwords, stemming
🔧 Feature Engineering: combine title, overview, genres, keywords, cast, crew
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(combined_features)
similarity_matrix = cosine_similarity(feature_matrix)
</details>
🤖 Model Training
<details> <summary>Click to Expand Model Training Steps</summary>
Train content-based filtering model:
# Run in Jupyter Notebook
jupyter notebook movie-recommendation-system.ipynb
</details>
⚙️ Hyperparameter Tuning
<details> <summary>Click to Expand Hyperparameter Tuning</summary>

Optimize TF-IDF vectorizer using GridSearchCV:
param_grid = {
    'max_features': [5000, 10000, 15000],
    'stop_words': ['english', None],
    'ngram_range': [(1,1), (1,2)],
    'min_df': [1,2,3]
}

grid_search = GridSearchCV(
    estimator=TfidfVectorizer(),
    param_grid=param_grid,
    scoring=make_scorer(custom_similarity_score),
    cv=3,
    n_jobs=-1
)
| Metric                                               | Score                                    |
| ---------------------------------------------------- | ---------------------------------------- |
| <span style="color:#6f42c1;">**Precision\@5**</span> | <span style="color:#28a745;">0.78</span> |
| <span style="color:#6f42c1;">**Recall\@5**</span>    | <span style="color:#28a745;">0.65</span> |
| <span style="color:#6f42c1;">**Diversity**</span>    | <span style="color:#28a745;">0.82</span> |
🎯 Evaluation Metrics
<div align="center">

🟣 Precision@5: 0.78

🔵 Recall@5: 0.65

🟢 Diversity: 0.82

</div>
🌐 Deployment
<details> <summary>Click to Expand Deployment Instructions</summary>

Local Deployment:
streamlit run app.py
</details>
📱 Usage

Select a movie from dropdown

Get top recommendations instantly

Click recommended movies for details

🔮 Future Improvements

🤝 Collaborative filtering

⚡ Hybrid recommendation system

👤 User preference integration

⏱️ Real-time updates

📊 More evaluation metrics

🙏 Acknowledgments

Dataset: TMDB 5000 Movie Dataset

Libraries: Scikit-learn, Pandas, Streamlit

Deployment: Render
