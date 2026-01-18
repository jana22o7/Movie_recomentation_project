import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load dataset
movies = pd.read_csv(r"C:\Users\User\Desktop\NM project\source code\movies.csv", low_memory=False)
movies = movies.head(1000)  # small subset


# Prepare tags
def convert(text):
    L = []
    try:
        for i in ast.literal_eval(text):
            L.append(i['name'])
    except:
        pass
    return " ".join(L)

movies['genres'] = movies['genres'].apply(convert)
movies['overview'] = movies['overview'].fillna('')
movies['genres'] = movies['genres'].fillna('')
movies['tags'] = (movies['overview'] + " " + movies['genres']).astype(str)

# Vectorizer (fit once)
cv = CountVectorizer(max_features=5000, stop_words='english')
cv.fit(movies['tags'])
vectors = cv.transform(movies['tags'])  # sparse matrix

# Recommendation function (compute similarity for one movie only)
def recommend(movie):
    try:
        index = movies[movies['title'] == movie].index[0]
        movie_vector = vectors[index]
        distances = cosine_similarity(movie_vector, vectors)[0]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        return [movies.iloc[i[0]].title for i in movie_list]
    except IndexError:
        return ["Movie not found"]
