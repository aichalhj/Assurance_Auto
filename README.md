# Tarification Automobile

Ce projet est une application **Streamlit** permettant d’analyser et de simuler la prime d’assurance automobile à partir des jeux de données simulées **freMTPL2freq** et **freMTPL2sev**, très utilisés en actuariat automobile.  
Le projet est disponible à l'adresse suivante : [https://assuranceauto.streamlit.app/](https://assuranceauto.streamlit.app/)

---

## Fonctionnalités

- Exploration interactive des données : fréquence et sévérité des sinistres selon différentes caractéristiques (véhicule, conducteur, région, zone, carburant, etc.)  
- Modélisation GLM pour la fréquence (Poisson) et la sévérité (Gamma)  
- Simulation de prime pure pour un contrat spécifique en saisissant les caractéristiques du conducteur et du véhicule  
- Visualisations interactives avec **Seaborn** et **Matplotlib**

---

## Modèle

- **Fréquence** : GLM Poisson  
- **Sévérité** : GLM Gamma  
- Les modèles sont entraînés sur un échantillon aléatoire pour rapidité  
- Les variables incluent : âge du véhicule, puissance, âge du conducteur, bonus-malus, marque, carburant, zone, région, densité  

---


## Installation et lancement

Cloner le dépôt, créer un environnement virtuel, installer les dépendances et lancer l’application Streamlit :  

```bash
git clone https://github.com/aichalhj/Assurance_Auto.git
cd Assurance_Auto
pip install -r requirements.txt
streamlit run App.py





