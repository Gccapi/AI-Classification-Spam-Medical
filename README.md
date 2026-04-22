# Intelligence Artificielle : Classification & Diagnostic

Ce projet, réalisé dans le cadre du cursus ingénieur à l'IPSA, explore l'application de modèles d'apprentissage supervisé à deux problématiques : la détection de courriels indésirables (Spams) et l'aide au diagnostic médical (Cancer du sein).

## Fonctionnalités

### Partie I : Détection de SPAM
- **Prétraitement (NLP) :** Nettoyage des données textuelles et vectorisation (CountVectorizer).
- **Modèles implémentés :** - **KNN (K-Nearest Neighbors)** : Classification basée sur la proximité.
  - **Naive Bayes** : Approche probabiliste.
  - **LDA (Linear Discriminant Analysis)** : Réduction de dimension et séparation linéaire.



### Partie II : Diagnostic du Cancer du Sein
- **Données :** Utilisation du dataset Scikit-Learn (features de noyaux cellulaires).
- **Algorithme :** Implémentation "from scratch" d'une **Régression Logistique** avec descente de gradient.
- **Analyse :** Étude de l'impact du nombre d'itérations sur l'accuracy du modèle.

## Résultats
Le projet met en évidence les compromis entre la complexité de calcul (notamment pour le LDA sur de grands jeux de données) et la précision des diagnostics. Les modèles atteignent des scores de performance élevés, démontrant l'efficacité des approches classiques avant l'usage de réseaux de neurones complexes.

## Installation & Usage
1. Cloner le dépôt :
   ```bash
   git clone [https://github.com/Gccapi/AI-Classification-Spam-Medical.git](https://github.com/Gccapi/AI-Classification-Spam-Medical.git)
   cd AI-Classification-Spam-Medical
2. Installer les dépendances :
   ```bash
   pip install numpy pandas scikit-learn matplotlib tqdm requests
3. Lancer les scripts :
   - python PartI_modif.py : Lance l'analyse et la détection de Spam.
   - python PartII_modif.py : Lance l'algorithme de diagnostic médical.

## Auteurs : 
   - Florian Alaux
   - Guilhem Perret-Bardou
  
