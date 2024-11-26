import numpy as np
import utils.matrix_operations as matOps
import scipy.linalg as scAlg


def Parametrice_Pose(T):
    """
    Calcula la parametrización de una matriz de transformación homogénea en rotación y traslación.
    
    - input:
        T: Matriz de transformación 4x4.
           T = | R  t |
               | 0  1 |
           R es la matriz de rotación (3x3), y t es el vector de traslación (3x1).
    
    - output:
        theta_rot: Lista con los tres parámetros de rotación (vector de ángulo-eje).
        tras: Lista con los dos ángulos (\\theta y \phi) que parametrizan la traslación.
    """
    # Extraemos la matriz de rotación R desde T
    R = T[0:3, 0:3]

    # Calculo de los parámetros de rotación (vector ángulo-eje)
    log_R = scAlg.logm(R)  # Devuelve una matriz anti-simétrica [omega]_x para pasar de R a su representación en el álgebra de Lie (so(3)).
    theta_rot = matOps.crossMatrixInv(log_R)  # Convierte la matriz cruzada a un vector (ángulo-eje)
    
    # Calculo de los parámetros de traslación
    theta_tras = np.arccos(np.clip(T[2, 2], -1, 1))  # Asegurarse de que el valor esté entre [-1, 1]
    if np.sin(theta_tras) > 1e-6:  # Evitar divisiones por cero
        phi_tras = np.arccos(np.clip(T[2, 0] / np.sin(theta_tras), -1, 1))
    else:
        phi_tras = 0  #(e.g., vector alineado con Z)
    
    tras = [theta_tras, phi_tras]

    # Paso 4: Devolver los resultados
    return theta_rot, tras


def ObtainPose(theta_rot, theta_tras, phi_tras):
    """
    Genera una matriz de transformación homogénea 4x4 a partir de los parámetros
    de rotación (vector ángulo-eje) y traslación (parametrizada con ángulos esféricos).
    
    - input:
        theta_rot: Vector de rotación (3x1 o 1x3).
        theta_tras: Ángulo de elevación (rad) para la traslación.
        phi_tras: Ángulo de azimut (rad) para la traslación.
    - output:
        T: Matriz de transformación homogénea 4x4.
    """
    # Calcular la matriz de rotación
    # Se usa el exponencial matricial para convertir el vector ángulo-eje en una matriz de rotación.
    R = scAlg.expm(matOps.crossMatrix(theta_rot))

    # Calcular el vector de traslación en parametrización esférica (theta_tras y phi_tras) a coordenadas cartesianas.
    Tras = np.array([
        np.sin(theta_tras) * np.cos(phi_tras),  # Componente X
        np.sin(theta_tras) * np.sin(phi_tras),  # Componente Y
        np.cos(theta_tras)                      # Componente Z
    ])

    T = np.hstack([R, Tras.reshape((3, 1))])
    T = np.vstack([T, [0, 0, 0, 1]])

    return T
