#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 4
#
# Title: Optical Flow
#
# Date: 22 November 2020
#
#####################################################################################
#
# Authors: Jose Lamarca, Jesus Bermudez, Richard Elvira, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import numpy as np
import cv2 as cv


def read_image(filename: str, ):
    """
    Read image using opencv converting from BGR to RGB
    :param filename: name of the image
    :return: np matrix with the image
    """
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def normalized_cross_correlation(patch: np.array, search_area: np.array) -> np.array:
    """
    Estimate normalized cross correlation values for a patch in a searching area.
    """
    # Complete the function
    i0 = patch
    i0_mean = np.mean(i0)
    i0_norm = i0 - i0_mean
    
    # Initialize the result matrix
    result = np.zeros(search_area.shape, dtype=np.float)
    margin_y = int(patch.shape[0]/2)
    margin_x = int(patch.shape[1]/2)

    # Moves the path along the search area
    for i in range(margin_y, search_area.shape[0] - margin_y):
        for j in range(margin_x, search_area.shape[1] - margin_x):
            i1 = search_area[i-margin_x:i + margin_x + 1, j-margin_y:j + margin_y + 1]
            i1_mean = np.mean(i1)
            i1_norm = i1 - i1_mean
            result[i, j] = np.sum(i0_norm * i1_norm) / (np.sqrt(np.sum(i0_norm ** 2)) * np.sqrt(np.sum(i1_norm ** 2)))
    
    return result


def seed_estimation_NCC_single_point(img1_gray, img2_gray, i_img, j_img, patch_half_size: int = 5, searching_area_size: int = 100):

    # Attention!! we are not checking the padding
    patch = img1_gray[i_img - patch_half_size:i_img + patch_half_size + 1, j_img - patch_half_size:j_img + patch_half_size + 1]

    # Define the searching area (limits up-left - low-right)
    i_ini_sa = i_img - int(searching_area_size / 2)
    i_end_sa = i_img + int(searching_area_size / 2) + 1
    j_ini_sa = j_img - int(searching_area_size / 2)
    j_end_sa = j_img + int(searching_area_size / 2) + 1

    # Obtenemos la region de busqueda de la imagen con los indices calculados
    search_area = img2_gray[i_ini_sa:i_end_sa, j_ini_sa:j_end_sa]
    
    # Llevamos a cabo la normalizacion de la imagen para el patch y el search_area definidos.
    result = normalized_cross_correlation(patch, search_area)
    print("\nResult: ", result)

    iMax, jMax = np.where(result == np.amax(result))

    # Calculamos el flujo optico
    # Explanation in text: The flow is calculated as the difference between the initial point
    # and the point where the maximum correlation is found. 
    i_flow = i_ini_sa + iMax[0] - i_img
    j_flow = j_ini_sa + jMax[0] - j_img

    return i_flow, j_flow

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    img1 = read_image("../data/frame10.png")
    img2 = read_image("../data/frame11.png")

    img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    # List of sparse points
    points_selected = np.loadtxt('../data/points_selected.txt')
    points_selected = points_selected.astype(int)

    template_size_half = 5
    searching_area_size: int = 15

    seed_optical_flow_sparse = np.zeros((points_selected.shape))
    for k in range(0,points_selected.shape[0]):
        i_flow, j_flow = seed_estimation_NCC_single_point(img1_gray, img2_gray, points_selected[k,1], points_selected[k,0], template_size_half, searching_area_size)
        seed_optical_flow_sparse[k,:] = np.hstack((j_flow,i_flow))

    print(seed_optical_flow_sparse)

