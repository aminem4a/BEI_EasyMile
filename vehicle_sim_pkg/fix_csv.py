import os
import pandas as pd
import numpy as np

def clean_file(input_name, output_name):
    base_dir = os.path.dirname(__file__)
    input_path = os.path.join(base_dir, "data", input_name)
    output_path = os.path.join(base_dir, "data", output_name)

    print(f"🔄 Traitement de : {input_name}...")

    if not os.path.exists(input_path):
        print(f"❌ Fichier introuvable : {input_path}")
        # On essaie de voir si le fichier "clean" existe déjà et s'il est corrompu, on le répare lui-même
        input_path = output_path
        if not os.path.exists(input_path):
            print("   Impossible de trouver une source.")
            return

    try:
        # 1. Lecture brute du fichier texte pour virer les points-virgules
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # REMPLACEMENT BRUT : on remplace tous les ; par des ,
        # Attention : si le fichier utilise la virgule pour les décimales (français), ça peut casser.
        # On suppose format US (point pour décimale).
        # Si format FR (virgule décimale), on remplace d'abord ',' par '.'
        
        # Stratégie robuste : 
        # a. Remplacer les ; par des , (séparateur)
        # b. Enlever les ; à la fin des lignes s'il y en a
        
        content = content.replace(';', ',')
        
        # Parfois on a ",," à cause du remplacement, on nettoie
        while ',,' in content:
            content = content.replace(',,', ',')

        # 2. Relecture avec Pandas via un buffer mémoire
        from io import StringIO
        df = pd.read_csv(StringIO(content), sep=',')

        # 3. Nettoyage des colonnes (strip)
        df.columns = [str(c).strip().lower() for c in df.columns]

        # 4. Forçage en numérique (au cas où il reste des textes bizarres)
        for col in df.columns:
            # On tente de convertir en nombre, les erreurs deviennent NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # On supprime les lignes qui ne contiennent que des NaN (lignes vides)
        df.dropna(how='all', inplace=True)
        # On remplace les NaN restants par 0
        df.fillna(0, inplace=True)

        # 5. Sauvegarde propre
        df.to_csv(output_path, index=False, sep=',')
        print(f"✅ Fichier réparé et sauvegardé : {output_name}")
        print(f"   Aperçu des colonnes : {list(df.columns)}")

    except Exception as e:
        print(f"🔥 Erreur critique lors du nettoyage : {e}")

if __name__ == "__main__":
    # Nettoie la map moteur
    # Note : Si tu n'as pas 'efficiency_map.csv' (l'original), change le nom ici
    # par 'efficiency_map_clean.csv' pour qu'il se nettoie lui-même.
    
    # Cas 1 : Tu as le fichier original sale
    if os.path.exists("data/efficiency_map.csv"):
        clean_file("efficiency_map.csv", "efficiency_map_clean.csv")
    
    # Cas 2 : Tu n'as que le fichier clean qui est en fait sale
    elif os.path.exists("data/efficiency_map_clean.csv"):
        clean_file("efficiency_map_clean.csv", "efficiency_map_clean.csv")
        
    else:
        print("Aucun fichier de map trouvé dans data/")