import os
import time
import pickle
import numpy as np
import pandas as pd
import functions_framework
from surprise import SVD
from google.cloud import storage
from collections import defaultdict


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
def get_top_n_art(predictions, deja_lus, n=5):
    """
    Renvoi les n articles les plus recommandées pour chaque utilisateur parmis ceux non lus
        @param predictions <Surprise Prediction> : prédictions issues de la méthode "test" d'un modèle Surprise
        @param deja_lus <list> : liste des articles déjà lus par l'utilisateur
        @param n <int> : nombre de recommandations à émettre
        @return <dict> : dictionnaire {user_id: [(category_id, predicted_rating), ...]}
    """
    # Map the predictions to each user
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        if iid not in deja_lus:
            top_n[uid].append((iid, est))

    # Sort the predictions for each user and retrieve the k highest ones
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def make_recommendation(model, user_id, art_by_user, art_df, n=5):
    """
    Renvoi un dataframe des articles les plus recommandés pour tous les utilisateur
        @param model <Surprise.SVD> : model entraîné servant à la prédiction
        @param user_id <int> : utilisateur pour qui effectuer la prédiction
        @param art_by_user <Pandas.DataFrame> : dataframe des interactions utilisateur/article
        @param art_df <Pandas.DataFrame> : dataframe listant les articles et leur catégorie
        @param n <int> : nombre d'articles à renvoyer (défaut : 5)
        @return <Pandas.DataFrame> : user_id, reco_art, reco_cat
    """
    # Constuction du dataset de prédiction
    article_list = art_by_user['article_id'].unique()
    nb_art = len(article_list)
    value_fill = np.full((nb_art,), 0)
    user_fill = np.full((nb_art,), user_id)
    art_rating = pd.DataFrame({
        'user_id': user_fill,
        'article_id': article_list,
        'value': value_fill
    })

    # Identifiation des articles lus
    user_art_list = art_by_user[art_by_user['user_id'] == user_id]['article_id'].values
    art_rating.loc[art_rating['article_id'].isin(user_art_list), 'value'] = 1

    # Récupération de la prédiction
    pred = model.test(art_rating.to_numpy())
    top_n = get_top_n_art(pred, user_art_list)

    # Préparation des données retournées
    pred_arts = [art for art, _ in top_n[user_id]]
    pred_cats = list(set(art_df[art_df['article_id'].isin(pred_arts)]['category_id'].values))
    cats = ", ".join([str(c) for c in list(set(pred_cats))])
    arts = ",".join([str(a) for a in pred_arts])

    return cats, arts


@functions_framework.http
def get_reco(request):
   """
   Reçoit une requête HTTP contenant un user_id et revoi une recommandation
       @param request <flask.Request> : l'objet de requête, request.args doit contenir "user_id"
       @return <JSON> {user_id, reco_cats, reco_arts, t_start, t_load, t_pred}
   """
   # Récupération du user_id
   request_json = request.json
   if request_json and 'user_id' in request_json:
       user_id = int(request_json['user_id'])

       # Chargement des fichier depuis le bucket du projet
       t_start = time.time()
       model = get_resource('svd_art.pkl', 'pkl')
       art_by_user = get_resource('articles_by_user.csv', 'csv')
       art_df = get_resource('articles_metadata.csv', 'csv')
       t_load = time.time()

       # Calcul des recommandations
       cats, arts = get_recommendations(model, user_id, art_by_user, art_df)
       t_pred = time.time()
       result = {'user_id': user_id, 'reco_cats': cats, 'reco_arts': arts, 't_start': t_start, 't_load': t_load, 't_pred': t_pred}
   
   # Si user_id n'est pas fourni
   else:
       result = {'user_id': '', 'reco_cats': '[]', 'reco_arts': '[]', 't_start': 0, 't_load': 0, 't_pred': 0}

   return result
