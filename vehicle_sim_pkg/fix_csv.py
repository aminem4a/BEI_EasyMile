import os

# Chemins (adaptez si besoin)
input_file = "data/efficiency_map.csv"
output_file = "data/efficiency_map_clean.csv"

if not os.path.exists(input_file):
    print(f"Erreur : Impossible de trouver {input_file}")
    exit()

print(f"Lecture de {input_file}...")

with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in:
    raw_content = f_in.read()

# --- LE NETTOYAGE ---
# 1. On enlève les crochets []
clean_content = raw_content.replace('[', '').replace(']', '')

# 2. On enlève les espaces inutiles autour du texte
clean_content = clean_content.strip()

# 3. On uniformise les sauts de ligne (pour éviter les lignes vides)
lines = [line.strip() for line in clean_content.splitlines() if line.strip()]
final_content = "\n".join(lines)

# 4. On sauvegarde le nouveau fichier
with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.write(final_content)

print("------------------------------------------------")
print(f"✅ SUCCÈS ! Fichier créé : {output_file}")
print("------------------------------------------------")
print("Aperçu des 5 premières lignes :")
print("\n".join(lines[:5]))
print("------------------------------------------------")