import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2 as cv



def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)


def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def plot_points_3D(ax, points, marker=".", color="b", size=10):
    """
    Dibuja puntos en un gráfico 3D.
    Parámetros:
    - ax: Objeto de ejes 3D de Matplotlib.
    - points: Array de puntos 3D (3xN).
    - marker: Marcador para los puntos (opcional).
    - color: Color de los puntos (opcional).
    - size: Tamaño de los puntos (opcional, por defecto 20).
    """
    ax.scatter(points[0, :], points[1, :], points[2, :], marker=marker, color=color, s=size)
    
def setup_3D_plot(x_label="X", y_label="Y", z_label="Z", equal_axis=True):
    """
    Configura un gráfico 3D.
    Parámetros:
    - fig_num: Número de figura (opcional, por defecto 1).
    - x_label: Etiqueta del eje X (opcional).
    - y_label: Etiqueta del eje Y (opcional).
    - z_label: Etiqueta del eje Z (opcional).
    - equal_axis: Si True, intenta igualar los ejes (opcional).
    Retorna:
    - ax: Objeto de ejes 3D configurado.
    """
    ax = plt.axes(projection="3d", adjustable="box")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    if equal_axis:
        # Bounding box to simulate equal axis scaling
        bounding_box = np.linspace(-1, 1, 2)
        ax.plot(bounding_box, bounding_box, bounding_box, "w.")

    return ax

def plot2DComparation(u_1,x1,image,imageName,figNum):
    """
    This function plots the 2D images.
    Inputs:
        u_1: 2d coordinates on the image
        x1: 2d coordinates on the image
        image: image
        imageName: image name
        figNum: figure number
    """
    fig1 = plt.figure(figNum)
    plt.imshow(image)
    plt.scatter(x1[0, :], x1[1, :], marker="x", c="r")
    plt.scatter(u_1[0, :], u_1[1, :], marker="o", c="b", s=6)
    plt.title(imageName)