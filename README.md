# INFO0911_medical_image_processing

## Table des matières

- **[Installation](#installation)**
- **[Utilisation](#utilisation)**
- **[Auteurs](#auteurs)**

## Installation

1. Déplacez-vous à la racine du projet
```bash
cd path/to/folder
```

2. Mettez en place l'environnement virtuel
```bash
python -m venv env
```

3. Activez l'environnement virtuel

Sous Windows :
```bash
env\Scripts\activate
```
Sous MacOS :
```bash
source env/bin/activate
```

4. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Afin de pouvoir exécuter le projet, il faut lancer la commande suivante à la racine du projet :
```bash
shiny run app.py
```

Une fois l'application exécutée, il faut aller à l'adresse suivante : *http://127.0.0.1:8000*.
Une fois dans l'application, il est possible de sélectionner différentes méthodes qui sont :
- Buitage
  - Gaussien
  - Sel & poivre
  - speckle
- Perona-Malik
- Coherence Enhancing
- Contour Detection

Leur ordre de sélection a une importance quant aux résultats obtenus. Pour pouvoir observer des résultats, il vaut mieux commencer par un bruitage et ensuite une autre fonction afin de voir concrètement leur impact sur des images non nettes.

### Score

Afin de comparer les différents résultats, 3 scores sont calculer. Ils sont les suivants :
- **PSNR [(Peak Signal to Noise Ratio)](https://fr.wikipedia.org/wiki/Peak_Signal_to_Noise_Ratio)** : Plus il est élevé, plus l'image est plus proche de l'image de référence.
- **MSE [(Mean squared error)](https://en.wikipedia.org/wiki/Mean_squared_error)** : Plus il est faible, moins il y aura de différences par rapport à l'image de référence.
- **SSIM [(Structural similarity index measure)](https://en.wikipedia.org/wiki/Structural_similarity_index_measure)** : Plus il est proche de 1, plus la structure de l'image est plus proche de celle de l'image de référence.

## Auteurs
- ARNOUDTS Kevin
- BAILLY Lucas
- DARVILLE Killian
- BRZYCHCY Loic
- LEHMAN Ylon
