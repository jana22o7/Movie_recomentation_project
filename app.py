from flask import Flask, render_template, request
from movie_recommender import recommend
import pandas as pd

app = Flask(__name__)

# Load movie titles for dropdown
movies = pd.read_csv(r"C:\Users\User\Desktop\NM project\source code\movies.csv", low_memory=False)
movie_titles = movies['title'].values

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []

    if request.method == "POST":
        movie_name = request.form.get("movie")
        recommendations = recommend(movie_name)

    return render_template("index.html", movies=movie_titles, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
