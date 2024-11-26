import numpy as np

def crossMatrixInv(M):
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return x


def transform_points(T, points):
    """
    Transforma un conjunto de puntos homogéneos (en columnas) de un marco de referencia a otro.
    
    Parámetros:
    - T: Matriz de transformación homogénea (4x4).
    - points: Matriz de puntos homogéneos (4 x N), donde cada columna es un punto.

    Retorno:
    - transformed_points: Matriz de puntos transformados (4 x N).
    """
    # Verifica que los puntos tengan dimensión compatible
    if points.shape[0] != 4:
        raise ValueError("Los puntos deben estar en coordenadas homogéneas (4xN).")

    # Multiplica la matriz de transformación por todos los puntos
    transformed_points = T @ points
    return transformed_points
