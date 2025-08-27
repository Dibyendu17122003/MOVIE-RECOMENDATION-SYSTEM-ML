<div align="center">

# 🎬 Movie Recommendation System
**Content-Based Movie Recommender with End-to-End ML Pipeline**  

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-green?logo=streamlit)](https://streamlit.io/) 
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikit-learn)](https://scikit-learn.org/) 

</div>

---

<div align="center">

<img src="assets/demo.gif" alt="Movie Recommendation Demo" width="700">

</div>

---

## ✨ Key Features

<div align="center">

<div style="display:flex; justify-content:center; flex-wrap: wrap; gap: 10px;">

<div style="background:#6f42c1; color:white; padding:12px 20px; border-radius:12px;">
🎯 Content-Based Filtering<br><sub>Cosine similarity recommendations</sub>
</div>

<div style="background:#28a745; color:white; padding:12px 20px; border-radius:12px;">
⚡ Hyperparameter Tuning<br><sub>GridSearchCV for TF-IDF</sub>
</div>

<div style="background:#17a2b8; color:white; padding:12px 20px; border-radius:12px;">
🔍 Advanced Preprocessing<br><sub>Stemming, stopwords, feature engineering</sub>
</div>

<div style="background:#ffc107; color:white; padding:12px 20px; border-radius:12px;">
💻 Interactive UI<br><sub>Streamlit frontend for easy usage</sub>
</div>

<div style="background:#fd7e14; color:white; padding:12px 20px; border-radius:12px;">
💾 Model Persistence<br><sub>Save/load similarity matrix</sub>
</div>

</div>
</div>

---

## 🛠️ Tech Stack

<div align="center">
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)] 
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-yellow?logo=scikit-learn)]
[![Pandas](https://img.shields.io/badge/Pandas-lightblue?logo=pandas)]
[![NLTK](https://img.shields.io/badge/NLTK-orange?logo=nltk)]
[![Streamlit](https://img.shields.io/badge/Streamlit-green?logo=streamlit)]
</div>

---

<div align="center" style="display:flex; flex-wrap: wrap; justify-content:center; gap:25px;"> <div style="background: linear-gradient(135deg, #6f42c1, #a855f7); color:white; padding:20px 25px; border-radius:15px; min-width:160px; text-align:center; box-shadow: 2px 6px 20px rgba(0,0,0,0.25); transition: transform 0.3s;"> 📄 <strong>app.py</strong><br> <sub>Streamlit frontend</sub> </div> <div style="background: linear-gradient(135deg, #28a745, #71f584); color:white; padding:20px 25px; border-radius:15px; min-width:160px; text-align:center; box-shadow: 2px 6px 20px rgba(0,0,0,0.25); transition: transform 0.3s;"> 📓 <strong>movie-recommendation-system.ipynb</strong><br> <sub>Jupyter Notebook</sub> </div> <div style="background: linear-gradient(135deg, #0d6efd, #66b2ff); color:white; padding:20px 25px; border-radius:15px; min-width:160px; text-align:center; box-shadow: 2px 6px 20px rgba(0,0,0,0.25); transition: transform 0.3s;"> 📦 <strong>setup.sh</strong><br> <sub>Install dependencies & NLTK</sub> </div> <div style="background: linear-gradient(135deg, #ffc107, #ffec7f); color:white; padding:20px 25px; border-radius:15px; min-width:160px; text-align:center; box-shadow: 2px 6px 20px rgba(0,0,0,0.25); transition: transform 0.3s;"> 🗂️ <strong>tmdb_5000_movies.csv</strong><br> <sub>Movies dataset</sub> </div> <div style="background: linear-gradient(135deg, #fd7e14, #ffb84d); color:white; padding:20px 25px; border-radius:15px; min-width:160px; text-align:center; box-shadow: 2px 6px 20px rgba(0,0,0,0.25); transition: transform 0.3s;"> 🗂️ <strong>tmdb_5000_credits.csv</strong><br> <sub>Credits dataset</sub> </div> </div> <style> div[align="center"] > div:hover { transform: scale(1.05); } </style>

---

## ⚡ Installation & Setup

<details>
<summary>Click to Expand</summary>

'bash'
# Clone repository
git clone https://github.com/yourusername/movie-recommendation-system.git
cd MOVIE-RECOMMENDATION-SYSTEM

# Run setup script (installs dependencies & NLTK)
bash setup.sh
Or manually:
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

python -c "import nltk; nltk.download('stopwords')"
</details>
📝 Data Preprocessing
<details> <summary>Click to Expand</summary>

Steps included:

✅ Data Cleaning: remove missing values & duplicates

📝 Text Processing: lowercase, remove stopwords, stemming

🔧 Feature Engineering: combine title, overview, genres, keywords, cast, crew
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(combined_features)
similarity_matrix = cosine_similarity(feature_matrix)
</details>
🤖 Model Training
<details> <summary>Click to Expand</summary>

Train content-based filtering model using Jupyter Notebook:
jupyter notebook movie-recommendation-system.ipynb
Explore data, preprocess, create feature matrix, compute similarity, save matrix.

</details>
⚙️ Hyperparameter Tuning
<details> <summary>Click to Expand</summary>

Optimize TF-IDF vectorizer:
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
</details>
🎯 Evaluation Metrics
<div align="center" style="display:flex; justify-content:center; gap:20px; flex-wrap: wrap;"> <div style="background: linear-gradient(135deg, #6f42c1, #a855f7); color:white; padding:15px 25px; border-radius:15px; min-width:120px; text-align:center; box-shadow: 2px 4px 10px rgba(0,0,0,0.2);"> 🟣 <strong>Precision@5</strong><br> <span style="font-size:20px;">0.78</span> </div> <div style="background: linear-gradient(135deg, #0d6efd, #66b2ff); color:white; padding:15px 25px; border-radius:15px; min-width:120px; text-align:center; box-shadow: 2px 4px 10px rgba(0,0,0,0.2);"> 🔵 <strong>Recall@5</strong><br> <span style="font-size:20px;">0.65</span> </div> <div style="background: linear-gradient(135deg, #198754, #4ade80); color:white; padding:15px 25px; border-radius:15px; min-width:120px; text-align:center; box-shadow: 2px 4px 10px rgba(0,0,0,0.2);"> 🟢 <strong>Diversity</strong><br> <span style="font-size:20px;">0.82</span> </div> </div>
🔹 Alternative GitHub Badge-Style (simpler, works in README)
<div align="center">






</div>
📱 Usage (Local)

Activate your virtual environment

Run Streamlit:
streamlit run app.py
Open browser → select a movie → get recommendations

Click on recommended movies for more details
🔮 Future Improvements

🤝 Collaborative filtering

⚡ Hybrid recommendation system

👤 Personalized recommendations

⏱️ Real-time updates

📊 More evaluation metrics

🙏 Acknowledgments

Dataset: TMDB 5000 Movie Dataset

Libraries: Scikit-learn, Pandas, NLTK, Streamlit

UI Inspiration: Modern Streamlit dashboards


