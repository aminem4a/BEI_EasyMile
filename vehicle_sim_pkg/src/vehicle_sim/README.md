# 🚗 Simulation d'Allocation de Couple (Torque Vectoring)

Ce projet, réalisé dans le cadre du PMP, simule différentes stratégies de répartition de couple pour un véhicule électrique à 4 moteurs-roues. L'objectif est d'optimiser l'efficacité énergétique globale (CosPhi) et la stabilité du véhicule en jouant sur la répartition avant/arrière.

## 📊 Stratégies Comparées

1.  **Inverse (Baseline) :** Répartition simple et équitable (50/50).
2.  **Piecewise :** Optimisation pure (cherche le meilleur rendement instantané, peut être brusque).
3.  **Smooth :** Optimisation avec contrainte de lissage (protège la mécanique).
4.  **Quadratic :** Minimisation des pertes Joules (robuste et stable).

---

## 🛠️ Installation

Suivez ces étapes scrupuleusement pour configurer l'environnement sur un nouveau PC.

### 1. Récupérer le projet
Ouvrez un terminal et lancez les commandes suivantes pour cloner le dépôt et entrer dans le dossier :

```bash
git clone <URL_DE_VOTRE_REPO_GIT>
cd vehicle_pkg

> ⚠️ **Important :** Ne sautez pas l'étape `cd vehicle_pkg`. Vous devez être à l'intérieur du dossier racine pour que les commandes fonctionnent.

### 2. Créer un environnement virtuel (Recommandé)
Cela isole le projet pour éviter les conflits de versions avec d'autres projets Python.

* **Sur Windows :**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
* **Sur Mac / Linux :**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
*(Vous devriez voir `(venv)` apparaître au début de la ligne de commande).*

### 3. Installer les dépendances
Installez les librairies mathématiques nécessaires :

```bash
pip install numpy pandas scipy matplotlib cvxpy

Voici la suite **exacte** du fichier `README.md`.

Tu as juste à copier le bloc ci-dessous et à le coller **à la suite** de ce que tu as déjà écrit (cela ferme le bloc de code `bash` que tu as ouvert et ajoute tout le reste).

---

```markdown

```

> ⚠️ **Important :** Ne sautez pas l'étape `cd vehicle_pkg`. Vous devez être à l'intérieur du dossier racine pour que les commandes fonctionnent.

### 2. Créer un environnement virtuel (Recommandé)

Cela isole le projet pour éviter les conflits de versions avec d'autres projets Python.

* **Sur Windows :**
```bash
python -m venv venv
.\venv\Scripts\activate

```


* **Sur Mac / Linux :**
```bash
python3 -m venv venv
source venv/bin/activate

```



*(Vous devriez voir `(venv)` apparaître au début de la ligne de commande).*

### 3. Installer les dépendances

Installez les librairies mathématiques nécessaires :

```bash
pip install numpy pandas scipy matplotlib cvxpy

```

---

## 🧹 Préparation des Données

Les données brutes issues des mesures expérimentales contiennent parfois des erreurs de formatage (points-virgules, texte...). Un script de nettoyage est inclus pour corriger cela automatiquement.

1. Vérifiez que le fichier `efficiency_map.csv` est bien présent dans le dossier `data/`.
2. Lancez le script de nettoyage à la racine du projet :

```bash
python fix_csv.py

```

✅ **Résultat :** Un fichier propre `data/efficiency_map_clean.csv` est généré. La simulation l'utilisera automatiquement.

---

## 🚀 Lancer les Simulations

Une fois installé, vous pouvez lancer deux types de tests depuis la racine du projet.

### A. Test de Validation (Performance Énergétique)

Ce script simule un roulage à vitesse stabilisée (ex: 13 km/h) pour calculer le gain d'énergie exact sur un scénario donné.

```bash
python examples/run_validation.py

```

**Ce que vous verrez :** Un tableau dans le terminal comparant l'énergie consommée (Wh), le CosPhi moyen et le gain en % par rapport à la méthode Inverse.

### B. Test de la Rampe (Preuve de Concept)

Ce script simule une montée progressive du couple (de 0 à 150 Nm) pour visualiser comment les stratégies réagissent dynamiquement. C'est idéal pour voir la différence de comportement entre les algorithmes.

```bash
python examples/run_ramp_test.py

```

**Ce que vous verrez :**

1. Un graphique de **Répartition** : L'Inverse reste plat (0.5), tandis que le Piecewise/Smooth saturent un essieu pour maximiser le rendement.
2. Un graphique de **Rendement** : L'impact des stratégies sur l'efficacité globale.

---

## ❓ Dépannage (FAQ)

**Q : J'ai une erreur `ModuleNotFoundError: No module named 'cvxpy'**`
R : Vous avez oublié d'installer les dépendances ou d'activer l'environnement virtuel. Refaites l'étape 2 et 3 de l'installation.

**Q : J'ai une erreur `FileNotFoundError**`
R : Vérifiez que vous lancez bien les commandes depuis la racine du dossier `vehicle_pkg` et pas depuis un sous-dossier (`src` ou `examples`).

**Q : Les résultats donnent 0% de gain ?**
R : À très faible charge (13 km/h à vide), le rendement moteur est très faible et plat partout ("zone morte" de la map). C'est normal. Lancez le script `run_ramp_test.py` pour voir les gains apparaître lors des phases d'accélération.

---

## 📁 Structure du Projet

* `src/` : Code source (Algorithmes d'allocation `allocation.py`, Modèles mathématiques `data_loader.py`).
* `data/` : Fichiers CSV (Cartographie moteur, Scénarios de conduite).
* `examples/` : Scripts de lancement (`run_validation.py`, `run_ramp_test.py`).
* `fix_csv.py` : Utilitaire de nettoyage de données.

```

```