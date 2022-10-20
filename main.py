import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import functions_framework
from surprise import SVD
from google.cloud import storage


storage_client = storage.Client()
bucket = storage_client.bucket("p9_models")


def get_resource(name, type):
    blob = bucket.blob(name)
    file = None
    with blob.open('rb') as f:
        if type == 'csv':
            file = pd.read_csv(f)
        elif type == 'pkl':
            file = pickle.load(f)
    return file


# Source : https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
def get_top_n_cat(predictions, n=10):
    """
    Renvoi les n catégories les plus recommandées pour chaque utilisateur
        @param predictions <Surprise Prediction> : prédictions issues de la méthode "test" d'un modèle Surprise
        @param n <int> : nombre de recommandations à émettre
        @return <dict> : dictionnaire {user_id: [(category_id, predicted_rating), ...]}
    """
    # Map the predictions to each user
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Sort the predictions for each user and retrieve the k highest ones
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def get_recommendations(user_id, top_n_cat, cat_by_user, art_by_user, art_df):
    """
    Renvoi une liste de 5 articles recommandés pour un utilisateur
        @param user_id <int> : ID de l'utilisateur pour lequel émettre des recommandations
        @param top_n_cat <list(int)> : top N catégories (avec poids associés) recommandées pour l'utilisateur
        @param cat_by_user <Panda.Dataframe> : Dataframe utilisé pour la recommandation si top_n_cat est vide
        @param art_by_user <Panda.Dataframe> : Dataframe des articles déjà lus et à ne pas reproposer
        @param articles_df <Panda.Dataframe> : Dataframe des articles par catégorie pour piocher les suggestions
        @return <list>, <list> : [articles recommandés], [catégories recommandées]
    """

    # Recherche des catégories recommandées pour l'utilisateur spécifié
    recommanded_cat = [cat for cat, _ in top_n_cat[user_id]]
    recommanded_wei = [wei for _, wei in top_n_cat[user_id]]
    
    # S'il n'y a pas de recommandation, nous utilisons les catégories historiques de l'utilisateur
    if not recommanded_cat:
        nb_cat = len(cat_by_user[cat_by_user['user_id'] == user_id])
        sorted_cats = cat_by_user[cat_by_user['user_id'] == user_id].nlargest(nb_cat, ['nb_clicks'])
        recommanded_cat = sorted_cats['category_id'].values
        recommanded_wei = sorted_cats['nb_clicks'].values
    
    # Soustraction des articles déjà lus par l'utilisateur
    deja_lu = art_by_user[art_by_user['user_id'] == user_id]['article_id'].values
    possible_art = art_df[~art_df['article_id'].isin(deja_lu)]
    
    # Selection des catégories à représenter en tenant compte des pondérations
    relative_wei = np.rint(np.divide(recommanded_wei, recommanded_wei[-1]))
    selected_cat = []
    for i in range(min(5, len(recommanded_cat))):
        selected_cat = np.concatenate((selected_cat, np.repeat(recommanded_cat[i], recommanded_wei[i])), axis=None)
    selected_cat = selected_cat.astype(int)[:min(5, len(selected_cat))]
    while len(selected_cat) < 5:
        selected_cat = np.concatenate(
            (selected_cat, selected_cat[0: min(len(selected_cat), 5 - len(selected_cat))]), axis=None)
    
    # Sélection des 5 articles non lus les plus populaires parmis les catégories recommandées avec pondération
    selected_art = []
    for c in selected_cat:
        pop_art = art_by_user[['user_id', 'article_id']].join(
            possible_art[['article_id', 'category_id']].set_index('article_id'), how='left', on='article_id')
        pop_art = pop_art.loc[
            (pop_art['category_id'] == c) & (~pop_art['article_id'].isin(selected_art)), ['article_id', 'user_id']]
        pop_art = pop_art.groupby('article_id').count().reset_index().sort_values(by='user_id', ascending=False)
        selected_art.append(int(pop_art.head(1)['article_id'].values))
    
    return list(set(selected_cat)), selected_art


@functions_framework.http
def get_reco(request):
   """
   Reçoit une requête HTTP contenant un user_id et revoi une recommandation
       @param request <flask.Request> : l'objet de requête, request.args doit contenir "user_id"
       @return <JSON> {user_id, reco_cats, reco_arts, t_start, t_load, t_pred}
   """
   # Récupération du user_id
   request_args = request.args
   if request_args and 'user_id' in request_args:
       user_id = request_args['user_id']

       # Chargement des fichier depuis le bucket du projet
       t_start = time.time()
       model = get_resource('svd.pkl', 'pkl')
       cat_by_user = get_resource('cat_rating_by_user.csv', 'csv')
       art_by_user = get_resource('articles_by_user.csv', 'csv')
       art_df = get_resource('articles_metadata.csv', 'csv')
       t_load = time.time()

       # Calcul des recommandations
       user_spec = cat_by_user[cat_by_user['user_id'] == user_id]
       user_pred = model.test(user_spec.to_numpy())
       top_n = get_top_n_cat(user_pred, 5)
       top_n_cat = get_top_n_cat(user_pred)
       cats, arts = get_recommendations(2, top_n_cat, cat_by_user, art_by_user, art_df)
       t_pred = time.time()
       result = {'user_id': user_id, 'reco_cats': json.dumps(cats), 'reco_arts': json.dumps(arts),
            't_start': t_start, 't_load': t_load, 't_pred': t_pred}
   
   # Si user_id n'est pas fourni
   else:
       result = {'user_id': '', 'reco_cats': '[]', 'reco_arts': '[]'}

   return result
