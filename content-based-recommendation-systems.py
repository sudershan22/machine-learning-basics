import pandas as pd
from tabulate import tabulate

#Download dataset : https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%205/data/moviedataset.zip

movies_df = pd.read_csv('moviedataset\movies.csv')
ratings_df = pd.read_csv('moviedataset\\ratings.csv')

movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df['genres'] = movies_df.genres.str.split('|')

moviesWithGenres_df = movies_df.copy()

for index, row in movies_df.iterrows():
    for genre in row['genres']:
        moviesWithGenres_df.at[index, genre] = 1

moviesWithGenres_df = moviesWithGenres_df.fillna(0)

ratings_df = ratings_df.drop('timestamp', 1)

#Content Bases Recomendation SYstem

userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ]
inputMovies = pd.DataFrame(userInput)

inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]

inputMovies = pd.merge(inputId, inputMovies)

inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

userMovies = userMovies.reset_index(drop=True)

userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])

genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

recommendationTable_df = ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())

recommendationTable_df = recommendationTable_df.sort_values(ascending=False)

print(tabulate(movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())], headers=['', 'movieId','title','genres','year']))


