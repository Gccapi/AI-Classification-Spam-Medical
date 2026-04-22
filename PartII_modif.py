
#---------------------------------PROJECT: PART II------------------------#

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer

from PartI_modif import LDA #pour éviter de copier coller la class ici 


data = load_breast_cancer() ;
X = data.data  #X représente les features du modèle
y = data.target #y sont les valeurs cibles qu'on veut prédire

# STANDARDIZE THE DATA
scale = StandardScaler() #standartScaler transforme nos données en faisant X-moyenne/écarttype (centre les données à 0 et met la variance à 1)
X = scale.fit_transform(X)  #comme dans la PI on normalise nos données

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # comme pour la PI on sépare a 0.8/0.2 entre train et test

#LOGISTIC REGRESSION

class LogisticRegressionCustom:
    def __init__(self, learning_rate, num_iterations):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))  #def ici une forme de calcul qui sera appliqué dans update_weights et predict en appelant la fonction sigmoid avec l'argument que l'on souhaite

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n) #crée un vecteur 0 
        self.bias = 0
        self.X = X
        self.y = y

        ##la boucle tournera autant de fois qu'il y a de nombres d'itérations, elle execute update_weights() à chaque fois    
        for i in range(self.num_iterations):  
            self.update_weights()

    def update_weights(self):
        #TO DO
        self.modele_lin = self.X@self.weights+self.bias
        y_pred = self.sigmoid(self.modele_lin) #on applique le calcul défini dans sigmoid sur modele_lin
        #on def les 2 gradients :
        dw = (1/self.m)*self.X.T@(y_pred- self.y) #poids
        db = (1/self.m)*np.sum(y_pred-self.y) #biais (np.sum fait la somme des valeurs d'un tableau sur un axe)
# on applique une itération
        self.weights -= self.learning_rate*dw # - parce qu'on réduit à chaque maj pour arriver à la meilleure coupe entre les classes
        self.bias -= self.learning_rate*db

    def predict(self,X):
        modele_lin = X@self.weights+self.bias #on le redef comme dans fit mais pour qu'il soit bien fait avec X_test ici  
        y_pred = self.sigmoid(modele_lin) #y_pred mais avec modele_lin calculé avec X_test
        Y = (y_pred >= 0.5).astype(int) #seuil pour classer défini ici à 0.5 (dans le sujet)
        return np.array(Y)


#TO DO : TEST THE REGRESSION LOGISTIC AND COMPARISION WITH LDA
#%% Question 11 avec accuracy score
print("Question 11 : Logistic Regression")
iterations = [1000,5000,10000,15000,20000,]
#on fait une boucle pour tester les différents nombres d'itérations 
for i in iterations : 
    #on appelle la méthode LogisticRegressionCustom et l'applique avec nos données, le learing_rate attendu et le nb d'itérations
    log_reg = LogisticRegressionCustom(learning_rate = 0.01, num_iterations = i)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print(f'\nNombres d\'itérations : {i}') #et on affiche les résultats dans la console
    print(f'Accuracy : {accuracy_score(y_test, y_pred):.4f}')

#%%Question 12 avec accuracy score

#rappelle LDA, importé de PartI et l'applique à ces nouvelles données
lda = LDA()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

print("\nQuestion 12 : comparaison avec LDA")
print(f'Accuracy : {accuracy_score(y_test, y_pred_lda):.4f}')
#%% graphe comparaison it
import matplotlib.pyplot as plt

iterations = [10,250,1000,2800,3000,5000,10000,11500,11800,15000,18000,20000]
acc = []

for i in iterations:
    log_reg = LogisticRegressionCustom(learning_rate=0.01, num_iterations=i)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    acc.append(accuracy_score(y_test, y_pred))

plt.plot(iterations, acc, marker='o')
plt.xlabel("Nombre d'itérations")
plt.ylabel("Accuracy sur X_test")
plt.title("Impact des itérations sur l'accuracy")
plt.grid()
plt.show()

#permet de visualiser l'évolution de l'accuracy avec le nombre d'itération
#%% Question 11 avec classification_report
print("Question 11 : Logistic Regression")
iterations = [1000,5000,10000,20000]
for i in iterations : 
    log_reg = LogisticRegressionCustom(learning_rate = 0.01, num_iterations = i)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print(f'Nombres d\'itérations : {i}')
    print(classification_report(y_test, y_pred))

#%%Question 12 avec classification_report

lda = LDA()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

print("\nQuestion 12 : comparaison avec LDA")
print(classification_report(y_test, y_pred_lda))

