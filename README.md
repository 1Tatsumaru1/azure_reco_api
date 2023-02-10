<h1 align="center">
  <br>
  <a href="http://www.amitmerchant.com/electron-markdownify">
    <img src="https://azure.microsoft.com/svghandler/functions/?width=600&height=315" alt="Markdownify" width="200">
  </a>
  <br>
  Recommendation API for Azure Functions
  <br>
</h1>

<h4 align="center">
  Built from this 
  <a href="https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom#clicks_sample.csv" target="_blank">Kaggle dataset</a>.
</h4>

![screenshot](https://github.com/1Tatsumaru1/azure_reco_api/blob/main/img/screenshot.png)

<p align="center">
  <a href="#description">Description</a> •
  <a href="#contents">Contents</a> •
  <a href="#credits">Credits</a> •
  <a href="#links">Links</a>
</p>

## Description

This project was part of my IA Engineering course at OpenClassrooms. 
The aim was to :<br>
* Build a recommendation model based on a <a href="https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom#clicks_sample.csv" target="_blank">behavioral dataset</a>
* Provide an API allowing the serving of the model through an Azure Function
* Building a font-end interface allowing users to interact with the model

## Contents

* **p9.ipynb** contains the complete analysis of the dataset and the building of 2 different approaches of the recommendation system
  - **Content-Based** : this approch is not really adapted to the data at hand, since there is no relevant metric to evaluate the particular interest of the user relative to an article. A more meta-study at the category level did not provide conclusive results either
  - **Collaborative-Filtering** : this is the one approach that provided the more consistent results here. The CF model is built through calculation of the cosine similarity among users, the major drawback of this technic being the cumputing time. Although a batch calculation on a more high-spec server could do the trick, for this project I contained the comparisons to 10k other users (which is also a viable solution for production, assuming the chosen user IDs are picked in a non-random fashion).
* **main.py** is the API main file, that needs to be uploaded together with **requirements.txt** in an Azure Function in order to issue prediction. The other files required in Azure storage are the model and the various intermediate CSV files, all of which can be retrieved by executing the Jupyter notebook

## Credits

This project makes use of the following open source packages:

- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-surprise](https://pypi.org/project/scikit-surprise/)

## Links

> [anthony.ledouguet.com](https://anthony.ledouguet.com) &nbsp;&middot;&nbsp;
> [GitHub](https://github.com/1Tatsumaru1) &nbsp;&middot;&nbsp;
> [@LinkedIn](https://www.linkedin.com/in/anthony-le-douguet/)
