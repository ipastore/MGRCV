from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
import torch
import os
import numpy as np
import itertools
import re

torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
print(f"Using device: {device}")

# Función para extraer el índice numérico de los nombres de las imágenes
def extract_index(filename):
    match = re.search(r"Img(\d{2})_", filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Formato incorrecto en nombre de archivo: {filename}")

# Pedir al usuario qué grupos incluir
# selected_groups = ["Group0", "Group1", "Group2", "Group3"]
selected_groups = ["Group1", "Group3"]

# selected_groups = ["Group1"]

if not selected_groups:
    raise ValueError("No se seleccionaron grupos válidos. Abortando.")

print(f"Procesando los siguientes grupos: {selected_groups}")

# Configuración de matcher y extractor
params_extractor = {"max_keypoints": 2048}
params_matcher = {"features": "superpoint", "max_layers": 12}

extractor = SuperPoint(**params_extractor).eval().to(device)
matcher = LightGlue(**params_matcher).eval().to(device)

# Crear carpetas de salida
output_path = "./matches_results"
os.makedirs(output_path, exist_ok=True)

# Generar combinaciones dentro de cada grupo y entre grupos
group_files = {group: [f for f in os.listdir(group) if f.endswith((".jpg", ".png"))] for group in selected_groups}

# Procesar combinaciones dentro de cada grupo
for group in selected_groups:
    group_output_path = os.path.join(output_path, f"{group}_{group}")
    os.makedirs(group_output_path, exist_ok=True)

    for image1, image2 in itertools.combinations(group_files[group], 2):
        # Determinar el orden basado en el índice
        index1 = extract_index(image1)
        index2 = extract_index(image2)
        if index1 > index2:
            image1, image2 = image2, image1

        print(f"Procesando (dentro del grupo {group}): {image1} vs {image2}")
        
        # Cargar imágenes
        img1 = load_image(os.path.join(group, image1))
        img2 = load_image(os.path.join(group, image2))
        
        # Extraer características
        feats1 = extractor.extract(img1.to(device))
        feats2 = extractor.extract(img2.to(device))
        matches12 = matcher({"image0": feats1, "image1": feats2})
        feats1, feats2, matches12 = [rbd(x) for x in [feats1, feats2, matches12]]
        
        # Extraer keypoints, matches y puntuaciones
        kpts1, kpts2 = feats1["keypoints"], feats2["keypoints"]
        matches = matches12["matches"]
        scores = matches12["scores"]
        
        # Guardar resultados en formato .npz
        output_file = os.path.join(group_output_path, f"{image1.split('.')[0]}_vs_{image2.split('.')[0]}_matches.npz")
        np.savez(output_file, keypoints0=kpts1, keypoints1=kpts2, matches=matches, scores=scores)
        print(f"Resultados guardados en: {output_file}")

# Procesar combinaciones entre grupos
for group1, group2 in itertools.combinations(selected_groups, 2):
    group_output_path = os.path.join(output_path, f"{group1}_{group2}")
    os.makedirs(group_output_path, exist_ok=True)

    for image1, image2 in itertools.product(group_files[group1], group_files[group2]):
        # Determinar el orden basado en el índice
        index1 = extract_index(image1)
        index2 = extract_index(image2)
        if index1 > index2:
            image1, image2 = image2, image1

        print(f"Procesando (entre {group1} y {group2}): {image1} vs {image2}")
        
        # Cargar imágenes
        img1 = load_image(os.path.join(group1, image1))
        img2 = load_image(os.path.join(group2, image2))
        
        # Extraer características
        feats1 = extractor.extract(img1.to(device))
        feats2 = extractor.extract(img2.to(device))
        matches12 = matcher({"image0": feats1, "image1": feats2})
        feats1, feats2, matches12 = [rbd(x) for x in [feats1, feats2, matches12]]
        
        # Extraer keypoints, matches y puntuaciones
        kpts1, kpts2 = feats1["keypoints"], feats2["keypoints"]
        matches = matches12["matches"]
        scores = matches12["scores"]
        
        # Guardar resultados en formato .npz
        output_file = os.path.join(group_output_path, f"{image1.split('.')[0]}_vs_{image2.split('.')[0]}_matches.npz")
        np.savez(output_file, keypoints0=kpts1, keypoints1=kpts2, matches=matches, scores=scores)
        print(f"Resultados guardados en: {output_file}")