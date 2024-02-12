import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

movies=pd.read_csv('data/movies.csv')
links=pd.read_csv('data/updated_links.csv')
ratings=pd.read_csv('data/ratings.csv')
tags=pd.read_csv('data/tags.csv')

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))




def string_to_array(s):
    number_strings = [num_str for num_str in s.strip('[]').split(' ') if num_str != '']
    # Convert each number string to float and create a NumPy array
    return np.array([float(num) for num in number_strings])

links['embedding']=links['embedding'].apply(string_to_array)

links_mo=pd.merge(links,movies[['movieId','title']],how='left',on='movieId')
links_mo

ratings_mo=pd.merge(ratings,movies[['movieId','title']],how='left',on='movieId')
ratings_mo

def create_mappings(dataframe_name):
    user2movie = dataframe_name.groupby('userId')['movieId'].unique().to_dict()
    movie2user = dataframe_name.groupby('movieId')['userId'].unique().to_dict()

    return user2movie, movie2user


def usermovierating_mappings(dataframe_name):

    usermovie2rating = dataframe_name.pivot_table(index='userId', columns='movieId', values='rating')
    
    usermovie2rating.fillna(0, inplace=True)
    

    return usermovie2rating


def get_rec_cf(movie_id, no_of_nearest_neighbors, usermovie_to_rating_train, movie_data,knn_model):
    # Find the nearest neighbors of the given movie
    movie_idx = usermovie_to_rating_train.columns.get_loc(movie_id)
    
    # Extract the movie vector for the given movie_id
    movie_vector = usermovie_to_rating_train.T.iloc[:, movie_idx].values.reshape(1, -1)
    
    # Use the movie vector to find the nearest neighbors
    distances, indices = knn_model.kneighbors(movie_vector, n_neighbors=no_of_nearest_neighbors + 1)
    
    # Flatten the arrays for easy handling
    distances = distances.flatten()
    indices = indices.flatten()
    
    # Exclude the first element of both arrays as it is the query movie itself
    distances = distances[1:]
    indices = indices[1:]
    
    # Map the indices to movie IDs
    similar_movies_ids = usermovie_to_rating_train.columns[indices].tolist()
    print(f'Similar movie IDs: {similar_movies_ids}')
    
    # Retrieve movie titles and prepare the recommendation dataframe
    cf_recs = []
    for i in range(len(similar_movies_ids)):
        similar_movie_id = similar_movies_ids[i]
        title = movie_data[movie_data['movieId'] == similar_movie_id]['title'].iloc[0]
        distance = distances[i]
        cf_recs.append({'Movie Id': similar_movie_id, 'Title': title, 'Distance': distance})
    
    # Create a dataframe from the recommendations list
    df = pd.DataFrame(cf_recs)
    
    return df


def cos_sim_rec(chosen_movie_id,links):
    cosine_sim=[]
    for i in range(len(links)):
        cosine_sim.append(cosine(links[links['movieId']==chosen_movie_id]['embedding'].values[0],links.loc[i]['embedding']))
    links_cos=links_mo.copy()
    links_cos['cosine sims']=cosine_sim
    
    return links_cos


def get_rec_cbr(recommended_movies,links_cos):
    rec_mov=pd.merge(recommended_movies,links_cos[['movieId','cosine sims']],how='left',left_on='Movie Id'
                 ,right_on='movieId')
    rec_mov=rec_mov.drop('movieId',axis=1)
    rec_mov=pd.merge(rec_mov,movies[['movieId','genres']],how='left',left_on='Movie Id'
                 ,right_on='movieId')
    rec_mov['rank']=(0.3*rec_mov['Distance'])+(0.7*rec_mov['cosine sims'])
    rec_mov=rec_mov.sort_values(by='rank', ascending=False)
    
    return rec_mov



usermovie_to_rating_train = usermovierating_mappings(ratings_mo)

knn_model = pickle.load(open('artifacts/knn_model.pkl','rb'))







# Example usage
"""chosen_movie_id = 305
recommended_movies = get_rec_cf(chosen_movie_id, 10, usermovie_to_rating_train, movies,knn_model)  # 'movies' should be your dataframe containing movie titles
print('Chosen movie based on movie id:', movies[movies['movieId'] == chosen_movie_id]['title'].iloc[0])
print("Recommended movies:")
print(recommended_movies)


links_cos=cos_sim_rec(chosen_movie_id,links)
links_cos_sort=links_cos.sort_values(by='cosine sims', ascending=False)

print('Movie Recommended on basis CBR')
print(links_cos_sort[['title','cosine sims']].loc[:10])

rec_mov=get_rec_cbr(recommended_movies,links_cos)

print('Movie Recommended on basis of CF & CBR')
print(rec_mov[['Title','rank']])
"""


def recommend_me_movies(chosen_movie_id, num, usermovie_to_rating_train,movies,knn_model,links):
    recommended_movies = get_rec_cf(chosen_movie_id, 10, usermovie_to_rating_train, movies,knn_model)  

    print('\n')
    print('\n')
    print('\n')
    print('Chosen movie based on movie id:', movies[movies['movieId'] == chosen_movie_id]['title'].iloc[0])
    print("Recommended movies on basis of CF")
    print(recommended_movies)
    print('\n')
    print('*'*100)
    print('\n')
    print('\n')

    
    links_cos=cos_sim_rec(chosen_movie_id,links)
    links_cos_sort=links_cos.sort_values(by='cosine sims', ascending=False)

    print('Movie Recommended on basis CBR')
    print(links_cos_sort[['title','cosine sims']].loc[:num])
    print('\n')
    print('*'*100)
    print('\n')
    print('\n')

    rec_mov=get_rec_cbr(recommended_movies,links_cos)

    print('Movie Recommended on basis of CF & CBR')
    print(rec_mov[['Title','rank']])

    print('\n')
    print('*'*100)
    print('\n')
    print('\n')

if __name__=="__main__":
    a=''
    while(a!='q'):
        a=input('Enter 1 to get recommendations or q to exit: ')
        
        if a=='1':
            
            chosen_movie_id=int(input('Enter movie id: '))
            num=int(input('Enter number of recommendations: '))

            recommend_me_movies(chosen_movie_id, num, usermovie_to_rating_train,movies,knn_model,links)
        
        else:
            break
