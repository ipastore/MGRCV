import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from lightglue import viz2d
from lightglue.utils import load_image
import sfm
import glob


def load_matches_data(npz_file):
    # Cargar datos desde el archivo .npz
    data = np.load(npz_file)
    keypoints0 = data['keypoints0']  # Keypoints de la primera imagen
    keypoints1 = data['keypoints1']  # Keypoints de la segunda imagen
    matches = data['matches']        # Matches entre ambas imágenes
    scores = data.get('scores', None)  # Scores opcionales
    matched_keypoints1 = keypoints0[matches[:, 0]]  # Índices de la primera columna de matches
    matched_keypoints2 = keypoints1[matches[:, 1]]  # Índices de la segunda columna de matches

    return matched_keypoints1, matched_keypoints2, keypoints0, keypoints1, matches, scores
    
def apply_RANSAC(image1_path, image2_path, matched_keypoints1, matched_keypoints2, nIter=1000, threshold=2, name_file='fundamental_matrix'):

    # Asegurarse de que el directorio existe
    os.makedirs('./results/fundamental', exist_ok=True)

    # Cargar la imagen desde la ruta imaginaria
    image_pers_1 = cv2.imread(image1_path, cv2.IMREAD_COLOR)
    image_pers_2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)

    # Convertir a coordenadas homogéneas
    matched_keypoints1_h = np.hstack((matched_keypoints1, np.ones((matched_keypoints1.shape[0], 1)))).T
    matched_keypoints2_h = np.hstack((matched_keypoints2, np.ones((matched_keypoints2.shape[0], 1)))).T

    # Llamar a RANSAC con los puntos homogéneos filtrados
    best_F, inliers_mask = sfm.ransac_fundamental_matrix(
        matched_keypoints1_h,  # Puntos homogéneos en la imagen 1
        matched_keypoints2_h,  # Puntos homogéneos en la imagen 2
        image1=image_pers_1,            # Opcional: pon tus imágenes aquí si las necesitas
        image2=image_pers_2,            # Opcional: pon tus imágenes aquí si las necesitas
        nIter=nIter,             # Número de iteraciones de RANSAC
        threshold=threshold         # Umbral para considerar un inlier
    )
    
    # Guardar la matriz fundamental en un archivo de texto
    np.savetxt(f'./results/fundamental/F_{name_file}.txt', best_F)
    
    return best_F, inliers_mask
    

def plot_keypoints(img1, img2, keypoints0, keypoints1, matches, name_image):
    
    # Asegurarse de que el directorio existe
    os.makedirs('./images/matches', exist_ok=True)
    
    # Extraer puntos clave emparejados
    m_kpts1 = keypoints0[matches[:, 0]]  # Keypoints emparejados en la imagen 1
    m_kpts2 = keypoints1[matches[:, 1]]  # Keypoints emparejados en la imagen 2
    
    # Visualización con viz2d
    axes = viz2d.plot_images([img1, img2])  # Mostrar las imágenes
    viz2d.plot_matches(m_kpts1, m_kpts2, color="lime", lw=0.2)  # Dibujar matches
    viz2d.add_text(0, f'Total Matches: {len(matches)}')  # Texto con información de matches
    plt.savefig(f'./images/matches/{name_image}_INLIERS.png')
    plt.close()  # Cerrar la figura para liberar memoria

def plot_inliers_RANSAC(img1, img2, inliers_mask, matched_keypoints1, matched_keypoints2, name_image):
    
    # Asegurarse de que el directorio existe
    os.makedirs('./images/inliers', exist_ok=True)
    
    inliers_kpts1 = matched_keypoints1[inliers_mask[:]]  # Keypoints emparejados en la imagen 1
    inliers_kpts2 = matched_keypoints2[inliers_mask[:]]  # Keypoints emparejados en la imagen 2
    
    # Visualización con viz2d para inliers
    axes = viz2d.plot_images([img1, img2])  # Mostrar las imágenes
    viz2d.plot_matches(inliers_kpts1, inliers_kpts2, color="blue", lw=0.2)  # Dibujar matches
    viz2d.add_text(0, f'Total INLIERS: {len(inliers_kpts1)}')  # Texto con información de matches
    plt.savefig(f'./images/inliers/{name_image}_INLIERS.png')
    plt.close()  # Cerrar la figura para liberar memoria
    
    
def plot_outliers_RANSAC(img1, img2, inliers_mask, matched_keypoints1, matched_keypoints2, name_image):
    
    # Asegurarse de que el directorio existe
    os.makedirs('./images/outliers', exist_ok=True)
    
    outliers_mask = ~inliers_mask
    outliers_kpts1 = matched_keypoints1[outliers_mask[:]]  # Keypoints emparejados en la imagen 1
    outliers_kpts2 = matched_keypoints2[outliers_mask[:]]  # Keypoints emparejados en la imagen 2
    
    # Visualización con viz2d para outliers
    axes = viz2d.plot_images([img1, img2])  # Mostrar las imágenes
    viz2d.plot_matches(outliers_kpts1, outliers_kpts2, color="red", lw=0.2)  # Dibujar matches
    viz2d.add_text(0, f'Total OUTLIERS: {len(outliers_kpts1)}')  # Texto con información de matches
    plt.savefig(f'./images/outliers/{name_image}_OUTLIERS.png')
    plt.close()  # Cerrar la figura para liberar memoria
    
# def save_matches_RANSAC(keypoints0, keypoints1, matches, inliers_mask, name_image):
#     # Asegurarse de que el directorio existe
#     os.makedirs('./results/inliers', exist_ok=True)

#     # Filtrar los keypoints y matches usando la máscara de inliers
#     inliers_keypoints0 = keypoints0[matches[inliers_mask, 0]]
#     inliers_keypoints1 = keypoints1[matches[inliers_mask, 1]]
#     inliers_matches = matches[inliers_mask]

#     # Guardar los keypoints y matches filtrados en un archivo .npz
#     np.savez(f'./results/inliers/{name_image}_inliers.npz', 
#                 keypoints0=inliers_keypoints0, 
#                 keypoints1=inliers_keypoints1, 
#                 matches=inliers_matches)
    
def save_matches_RANSAC(keypoints0, keypoints1, matches, inliers_mask, name_image):
    # Asegurarse de que el directorio existe
    os.makedirs('./results/inliers', exist_ok=True)

    # # Filtrar los keypoints y matches usando la máscara de inliers
    # inliers_keypoints0 = keypoints0[matches[inliers_mask, 0]]
    # inliers_keypoints1 = keypoints1[matches[inliers_mask, 1]]
    # inliers_matches = matches[inliers_mask]

    # Guardar los keypoints y matches filtrados en un archivo .npz
    np.savez(f'./results/inliers/{name_image}_inliers.npz', 
                keypoints0=keypoints0, 
                keypoints1=keypoints1, 
                matches=matches,
                inliers_matches = matches[inliers_mask])



if __name__ == "__main__":
    
    path = '../Images/Set_12MP/matches_results/Group1_Group3'
    if not os.path.exists(path):
        print(f"Error: El directorio {path} no existe.")
    else:
        print(f"Archivos en {path}:")
        print(os.listdir(path))
    
    npz_files = glob.glob(f'{path}/*.npz')
    for npz_file in npz_files:
        base_name = os.path.basename(npz_file)  # Extrae el nombre del archivo (sin el directorio)
        print(f"Processing {base_name}...")
        img1_name, img2_name = base_name.split('_vs_')  # Divide los nombres de las imágenes usando '_vs_' como separador
        img1_name = img1_name.replace('_matches.npz', '')  # Elimina '_matches.npz' del nombre de la primera imagen
        img2_name = img2_name.replace('_matches.npz', '')  # Elimina '_matches.npz' del nombre de la segunda imagen

        image1_path = f"../Images/Set_12MP/EntireSet/{img1_name}.jpg"
        image2_path = f"../Images/Set_12MP/EntireSet/{img2_name}.jpg"

        img1 = load_image(image1_path)
        img2 = load_image(image2_path)
        
        if img1 is None or img2 is None:
            print(f"Error: One of the images {image1_path} or {image2_path} could not be loaded.")
            continue

        matched_keypoints1, matched_keypoints2, keypoints0, keypoints1, matches, scores = load_matches_data(npz_file)

        best_F, inliers_mask = apply_RANSAC(
            image1_path,          # Ruta de la primera imagen
            image2_path,          # Ruta de la segunda imagen
            matched_keypoints1,   # Puntos clave emparejados en la primera imagen
            matched_keypoints2,   # Puntos clave emparejados en la segunda imagen
            nIter=1000,           # Número de iteraciones de RANSAC
            threshold=4,           # Umbral de distancia para considerar un punto como inlier
            name_file = f"{img1_name}_vs_{img2_name}"  # Nombre del archivo de la matriz fundamental
        )

        plot_keypoints(img1, img2, keypoints0, keypoints1, matches, f"{img1_name}_vs_{img2_name}")
        plot_inliers_RANSAC(img1, img2, inliers_mask, matched_keypoints1, matched_keypoints2, f"{img1_name}_vs_{img2_name}")
        plot_outliers_RANSAC(img1, img2, inliers_mask, matched_keypoints1, matched_keypoints2, f"{img1_name}_vs_{img2_name}")
        save_matches_RANSAC(keypoints0, keypoints1, matches, inliers_mask, f"{img1_name}_vs_{img2_name}")
 
    
    
    
    