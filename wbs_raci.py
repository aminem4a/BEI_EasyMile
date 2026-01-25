import pandas as pd
from datetime import datetime

# --- 1. DONNÉES DU PROJET ---

tasks = [
    # Phase 1
    {"Activite": "Recherche méthodes & Stratégie allocation", "Debut": "04/11/2025", "Fin": "12/11/2025", "Qui": "Mouhsine, Amine E"},
    {"Activite": "Modélisation diagramme de classes (UML)", "Debut": "04/11/2025", "Fin": "12/11/2025", "Qui": "Diego, Amine M"},
    # Phase 2
    {"Activite": "Correction UML & Implémentation Python base", "Debut": "19/11/2025", "Fin": "24/11/2025", "Qui": "Diego, Amine M"},
    {"Activite": "Modélisation moteur asynchrone & ID Matlab", "Debut": "19/11/2025", "Fin": "24/11/2025", "Qui": "Mouhsine, Amine E"},
    {"Activite": "Test modèle identifié (Boucle Ouverte)", "Debut": "03/12/2025", "Fin": "03/12/2025", "Qui": "Mouhsine, Amine E"},
    {"Activite": "Tests scénarios Python (charge/montée)", "Debut": "03/12/2025", "Fin": "03/12/2025", "Qui": "Diego, Amine M"},
    # Phase 3
    {"Activite": "Redéfinition Objectifs & Contraintes", "Debut": "10/12/2025", "Fin": "17/12/2025", "Qui": "Mouhsine, Amine E"},
    {"Activite": "Optimisation Énergie/Couple/Cos(phi)", "Debut": "10/12/2025", "Fin": "17/12/2025", "Qui": "Diego, Amine M"},
    # Phase 4 (Janvier)
    {"Activite": "Stratégie Couple Négatif (Freinage)", "Debut": "05/01/2026", "Fin": "14/01/2026", "Qui": "Amine E"},
    {"Activite": "Migration code Identification (Matlab->Python)", "Debut": "05/01/2026", "Fin": "14/01/2026", "Qui": "Mouhsine"},
    {"Activite": "Définition Bloc Véhicule & Comparaison", "Debut": "05/01/2026", "Fin": "14/01/2026", "Qui": "Diego"},
    {"Activite": "Simulation Boucle Ouverte (Système complet)", "Debut": "05/01/2026", "Fin": "14/01/2026", "Qui": "Amine M"},
    {"Activite": "Simulation Boucle Fermée (FINAL)", "Debut": "20/01/2026", "Fin": "20/01/2026", "Qui": "Amine M"},
]

# --- 2. TRAITEMENT POUR LE FORMAT GANTT EXCEL ---
# On calque le format sur ton fichier "Diagramme_Gantt_BEI...csv"
gantt_rows = []

for t in tasks:
    d_start = datetime.strptime(t["Debut"], "%d/%m/%Y")
    d_end = datetime.strptime(t["Fin"], "%d/%m/%Y")
    # Calcul de la durée en jours (+1 pour inclure le jour de fin)
    duration = (d_end - d_start).days + 1
    
    # Formatage des dates pour Excel (AAAA-MM-JJ est souvent plus sûr, ou garder format FR)
    gantt_rows.append({
        "ACTIVITÉ": t["Activite"],
        "DÉBUT DU PLAN": t["Debut"],
        "DURÉE DU PLAN": duration,
        "DÉBUT RÉEL": "",       # Laisser vide comme dans un template vierge
        "DURÉE RÉELLE": "",
        "POURCENTAGE ACCOMPLI": 0,
        "ATTRIBUE A": t["Qui"]
    })

df_gantt = pd.DataFrame(gantt_rows)

# --- 3. TRAITEMENT POUR LE WBS (HIERARCHIQUE) ---
wbs_data = [
    {"Niveau": "1", "Tache": "Conception & État de l'art", "Livrable": "Rapport & UML"},
    {"Niveau": "1.1", "Tache": "Recherche & Stratégie", "Livrable": "État de l'art"},
    {"Niveau": "1.2", "Tache": "Modélisation Système", "Livrable": "Diagramme de classes"},
    {"Niveau": "2", "Tache": "Modélisation Physique", "Livrable": "Modèle Moteur"},
    {"Niveau": "2.1", "Tache": "Identification & Migration", "Livrable": "Code Python"},
    {"Niveau": "3", "Tache": "Développement & Optimisation", "Livrable": "Algorithmes"},
    {"Niveau": "3.1", "Tache": "Implémentation & Tests", "Livrable": "Courbes opti"},
    {"Niveau": "4", "Tache": "Simulation & Validation", "Livrable": "Résultats finaux"},
]
df_wbs = pd.DataFrame(wbs_data)

# --- 4. TRAITEMENT POUR LE RACI ---
raci_data = [
    {"Tache": "Stratégie & Recherche", "Mouhsine": "R", "Amine E": "R", "Diego": "I", "Amine M": "I"},
    {"Tache": "Modélisation UML", "Mouhsine": "I", "Amine E": "I", "Diego": "R", "Amine M": "R"},
    {"Tache": "Code Python & Opti", "Mouhsine": "I", "Amine E": "I", "Diego": "R", "Amine M": "R"},
    {"Tache": "Identification Moteur", "Mouhsine": "R", "Amine E": "R", "Diego": "I", "Amine M": "I"},
    {"Tache": "Simulations Finales", "Mouhsine": "I", "Amine E": "I", "Diego": "C", "Amine M": "R"},
]
df_raci = pd.DataFrame(raci_data)

# --- 5. EXPORT CSV (Compatible Excel français) ---
print("Génération des fichiers...")

# Encodage utf-8-sig pour gérer les accents correctement dans Excel
df_gantt.to_csv("Format_GANTT_Import.csv", index=False, sep=";", encoding="utf-8-sig")
df_wbs.to_csv("Format_WBS_Rapport.csv", index=False, sep=";", encoding="utf-8-sig")
df_raci.to_csv("Format_RACI_Rapport.csv", index=False, sep=";", encoding="utf-8-sig")

print("Terminé ! 3 fichiers CSV ont été créés.")
print("1. Ouvre 'Format_GANTT_Import.csv'.")
print("2. Copie les colonnes et colle-les dans ton fichier Excel 'Diagramme_Gantt_BEI...'.")