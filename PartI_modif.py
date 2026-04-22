# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 15:29:24 2025

@author: floal
"""

#---------------------------------PROJECT: PART I---------------------#

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,euclidean_distances
import zipfile #il y avait une double importation de numpy nous l'avons enlevé
import requests
from io import BytesIO

from tqdm import tqdm #pour l'ajout d'une barre de progression (exec trop longue durant les tests LDA avec parfois plus de 10minutes à tourner pendant les tests sans savoir si c'est planté ou en calcul)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
response = requests.get(url) #Va chercher le fichier de url
zip_file = zipfile.ZipFile(BytesIO(response.content)) #récupère le fichier zip en créant un objet pour travailler dessus + BytesIO convertis les données en format qu'on peut lire

# Extraire le fichier SMSSpamCollection
with zip_file.open('SMSSpamCollection') as file: #ouvre le zip
    df = pd.read_csv(file, sep='\t', names=['label', 'message'])
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.to_csv('SMSSpamCollection_processed.csv', index=False)

#TO DO: PRE-PROCESS THE DATA
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message']).toarray() #normalise et transforme ne matrice les données
y = df['label'].values
vocabulary = vectorizer.get_feature_names_out()
#print(sum(X)) #on vérifie que la matrice est non nulle, plus utile maintenant 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42) #on separe les données entre train et test avec une proportion 0.8/0.2

# KNN
class KNN:
    def __init__(self, k=3):
        self.k = k
   
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
   
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
   
    def _predict(self, x):
        distances = euclidean_distances(self.X_train, [x]).flatten()    #Calcul de distance grace à euclidean_distances importé + converti en tableau 1D avec .flatten()
        indices_proches = np.argsort(distances)[:self.k]    #classe en ordre croissant et garde des k premiers (donc les plus proches)
        labels_proches = self.y_train[indices_proches]    #on garde les labels des k voisins
        return np.bincount(labels_proches).argmax()     #compte chaque classe et ressort celle qui apparait le plus



#NAIVEBAYES

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)
        #on définit 3 matrices de 0 pour pouvoir les remplir ensuite 
        self._priors = np.zeros(n_classes)
        self._means = np.zeros((n_classes,n_features))
        self._variances = np.zeros((n_classes, n_features))
       
        for idx, c in enumerate(self._classes):
            X_c = X[y == c]   #on garde que les lignes de X de classe c
            N_k = X_c.shape[0]  
            #les 3 estimateurs de la question 8 :
            self._priors[idx] = N_k/n_samples  #pi_k
            self._means[idx] = np.mean(X_c,axis=0) #mu_k,p  (np.mean renvoie la moyenne sur un axe)
            self._variances[idx] = np.var(X_c,axis=0)+ 1e-6  #sigma_k,p² + petite valeur pour éviter les problèmes num en étant trop proche de 0 
                                                                #1e-6 prit au car on le retrouve en param pour LDA, peut être modif pour autre nb proche de 0
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
   
    def _predict(self, x):
        posteriors = []
       
        for idx, c in enumerate(self._classes):     #sépare en 2 termes y* parce qu'on nous donne "posteriors = []"
            prior = np.log(self._priors[idx])  #log(pi_k) 
            lvg = -0.5 * np.sum((x-self._means[idx])**2/self._variances[idx])  # log vraissemblance gauss
            posteriors.append(prior + lvg)  #additionne les 2 log pour avoir le score de la classe c 
            
        return self._classes[np.argmax(posteriors)]   # retourne la classe avec le plus grand score
   

   

#LDA

class LDA:
    def __init__(self, param = 1e-6):
        self.param = param
       
    def fit(self, X, y):
        self.classes = np.unique(y) #np.unique renvoie les entrées du tableaux dans l'ordre où elles apparaissent
        n_features = X.shape[1]
        self.means = np.zeros((len(self.classes), n_features))
        self.priors = np.zeros(len(self.classes))
        self.cov = np.zeros((n_features, n_features))
       
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]   #comme pour Bayes, on garde ce qui est de classe c
            #les 3 estimateurs : 
            self.means[idx] = np.mean(X_c,axis=0)  # mu_k 
            self.priors[idx] = X_c.shape[0]/X.shape[0]  # pi_k
        self.cov = np.cov(X.T)+np.eye(n_features)*self.param  # Sigma_k + I*param  #np.cov(X.T) fait la covariance et return une mat (P,P), permet de voir la corrélation entre les données
        #pour être sûr de pouvoir l'inverser dans _predict on ajoute la matrice identité*param pour avoir des valeurs proches de 0 sur la diag 
        #-> assure des valeurs propres strict positives donc matrice inversible
        #-> évite un det trop proche de 0 (Sigma est censée etre inversible en théorie)
        
        #on déplace l'inversion ici pour avoir la progression de _predict correcte et bien voir la fin du calcul d'inverse
        print("\nInversion de la matrice de covariance en cours...")
        self.inv_cov = np.linalg.inv(self.cov) #on inverse Sigma_k 
        print("Inversion terminée") 
    
    def predict(self, X):
        y_pred = []
        for x in tqdm(X, desc="Prédiction LDA en cours"): #la barre de progression pour calmer ma patience pendant les nombreux tests qui semblaient si long  
            y_pred.append(self._predict(x)) #rempli y_pred avec la valeur trouvée en executant _predict 
        return np.array(y_pred)
   
    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes): #pareil que NaiveBayes, on sépare y* ici 
            quad = -0.5*(x-self.means[idx]).T@self.inv_cov@(x-self.means[idx]) #@ donne le produit matriciel
            prior = np.log(self.priors[idx])
            posteriors.append(quad+prior) #on somme pour avoir posteriors comme dans NaiveBayes
        return self.classes[np.argmax(posteriors)]

    

#%%
#TEST OF THE ALGORITHMS

# REMARQUE : lancer la dernière case testera toutes les méthodes d'un coup
# Question 6 Eval KNN
if __name__ == "__main__":  #pour éviter l'exec quand on l'appelle dans PartII

    tests_sizes = [0.9, 0.7, 0.5, 0.2] #chaque boucle est longue, on peut laisser que 0,2 si on veut juste tester (mais la question 6 demande cette comparaison)
    print('Question 6 : KNN')
    for s in tests_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=s,random_state=42) #on rappelle et sépare les données ici, pas nécessaire comme fait au début mais rassurant  
        #on appelle la méthode KNN def plus haut et l'applique à nos données
        knn = KNN(k=5)
        knn.fit(X_train,y_train)
        y_pred = knn.predict(X_test)
        
        acc = accuracy_score(y_test,y_pred)
        print(f'Taille du jeu d\'entraînement: {int(len(X_train))} - Accuracy: {acc:.4f}')  #.4f pour laisser 4 chiffres après la virgule
        
# on pouvait s'en douter, plus la taille du jeu d'entrainement est grande plus la précision augmente 
#(on passe de acc = 0.926 pour 80% à 0.869 pour 10%)
# méthode lente (~3min pour la première boucle, puis jusqu'à ~5min)

#%% Question 12 Eval Naive Bayes (classification_report, version test des 3 méthode avec accuracy_score après)
if __name__ == "__main__":
#on appelle la méthode NaiveBayes def plus haut et l'applique à nos données
    nb = NaiveBayes()
    nb.fit(X_train,y_train)
    y_pred = nb.predict(X_test)

    print('Question 12 : Naive Bayes')
    print(classification_report(y_test,y_pred)) #renvoie un tableau détaillé avec precision, rappel, f1-score et accuracy 

#plus précis en ham que scam, logique car plus de données en ham 
#méthode très rapide (~3seconde)
#%% Question 18 Eval LDA (classification_report)
if __name__ == "__main__":
#on appelle la méthode LDA et l'applique à nos données 
    lda = LDA()
    lda.fit(X_train,y_train)
    y_pred = lda.predict(X_test)

    print('Question 18 : LDA')
    print(classification_report(y_test,y_pred))
    
#méthode lente (~2minutes)
#%% comparaison des 3 avec les tableaux classification_report
if __name__ == "__main__":

    modeles = {'KNN': KNN(k=5),'Naive Bayes': NaiveBayes(),'LDA': LDA()} 

    for nom, modele in modeles.items(): #boucle qui exectute les 3 modeles pour sortir le tableau detaillé 
        modele.fit(X_train,y_train)
        y_pred = modele.predict(X_test)
        print(f'\n-- {nom} --')
        print(classification_report(y_test,y_pred))
        
#~8min
#%% Test avec accuracy_score si on veut pas lire le tableau complet de classification_report

if __name__ == "__main__":  
    
    modeles = {'KNN': KNN(k=5),'Naive Bayes': NaiveBayes(),'LDA': LDA()}

    for nom, modele in modeles.items():  #boucle qui execute les 3 modeles pour afficher l'accuracy_socre
        modele.fit(X_train,y_train)
        y_pred = modele.predict(X_test)
        print(f'\n-- {nom} --')
        print(f'Accuracy : {accuracy_score(y_test,y_pred):.4f}')
        
#%%  hors question, nuages de points 

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

# Réduction de dimension avec PCA sur X_train pour entraîner les modèles
pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# Création d'une fonction pour afficher les prédictions

def plot_predictions(model, X_train, X_test, y_test, title):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(8,6))
    colors = ['blue' if label == 0 else 'red' for label in y_pred]
    plt.scatter(X_test_reduced[:, 0], X_test_reduced[:, 1], c=colors, alpha=0.6, edgecolors='k')
    plt.xlabel('Composante principale 1')
    plt.ylabel('Composante principale 2')
    plt.title(title)
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Ham (0)'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Spam (1)')]
    plt.legend(handles=legend_elements)
    plt.show()

# Affichage des nuages de points pour chaque modèle
plot_predictions(KNN(k=5), X_train, X_test, y_test, 'Nuage de points des prédictions KNN')
plot_predictions(NaiveBayes(), X_train, X_test, y_test, 'Nuage de points des prédictions Naive Bayes')
plot_predictions(LDA(), X_train, X_test, y_test, 'Nuage de points des prédictions LDA')



