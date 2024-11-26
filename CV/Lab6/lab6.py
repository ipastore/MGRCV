from utils.matrixOperationsCV import *
from utils.NCCTemplate import *
from utils.interpolationFunctionsCV import *


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
        

def main():
        
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)
    
    ### EXERCICE 2.1: Kannala-Brandt Model ###
    
    if EXERCICE_2_1:
        seed_optical_flow_sparse = exercise_2_1()

    ### EXERCICE 2.2: Triangulation ###
    
    if EXERCICE_2_2:
        x_A_w = exercise_2_2(seed_optical_flow_sparse)
        
    if EXERCICE_2_3:
        x_A_w = exercise_2_3()
        
    if EXERCICE_2_4:
        x_A_w = exercise_2_4()
     
    ### EXERCICE 3: Bundle adjustment ###
    
    if EXERCICE_3:
        exercise_3()
                     
def exercise_2_1():
    
    print("\n*************************************************************************************")
    print("* EXERCICE 2.1: Motion  by using Normalized Cross Correlation(NCC) brute-force search *")
    print("*************************************************************************************\n")
    
    # We load the images
    img1 = read_image("./data/frame10.png")
    img2 = read_image("./data/frame11.png")
    img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    # List of sparse points selected
    points_selected = np.loadtxt('./data/points_selected.txt')
    points_selected = points_selected.astype(int)

    # Define the template size and the searching area size
    template_size_half = 5          # Define a template of 5x5 pixels. This patch will move around the searching area defined by the searching_area_size
    searching_area_size: int = 15   # Define a region of 15x15 pixels around the pixel to search for the best match using the template patch

    seed_optical_flow_sparse = np.zeros((points_selected.shape))
    for k in range(0,points_selected.shape[0]):
        i_flow, j_flow = seed_estimation_NCC_single_point(img1_gray, img2_gray, points_selected[k,1], points_selected[k,0], template_size_half, searching_area_size)
        seed_optical_flow_sparse[k,:] = np.hstack((j_flow,i_flow))

    print(seed_optical_flow_sparse)
    
    return seed_optical_flow_sparse

def exercise_2_2(seed_optical_flow_sparse):
    
    print("\n*******************************************************************************")
    print("* EXERCICE 2.2: Lucas Kanade approach that refines your NCC brute-force search *")
    print("*******************************************************************************\n")
    
    # We load the images
    img1 = read_image("./data/frame10.png")
    img2 = read_image("./data/frame11.png")
    img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    
    
    # Define the template size and the searching area size
    patch_half_size = 5          # Define a template of 5x5 pixels. This patch will move around the searching area defined by the searching_area_size
    searching_area_size: int = 15   # Define a region of 15x15 pixels around the pixel to search for the best match using the template patch


    # List of sparse points selected
    points_selected = np.loadtxt('./data/points_selected.txt')
    points_selected = points_selected.astype(int)
    points_for_gradient = points_selected[:, [1, 0]]  # Swap the columns

    # Compute the gradients of the image [y,x]
    gradients_j0 = numerical_gradient(img1_gray, points_for_gradient)
    debug_print("gradients_j0: ", gradients_j0, sep="\n")
    
    
    punto = 0
    
    # Compute the matrix elements of A
    A_11 = np.sum(gradients_j0[punto, 1] ** 2)
    A_12 = np.sum(gradients_j0[punto, 1] * gradients_j0[punto, 0])
    A_22 = np.sum(gradients_j0[punto, 0] ** 2)
    A = np.array([[A_11, A_12], [A_12, A_22]])
    
    # We check that the matrix A is not singular and consequently we can compute its inverse
    if np.linalg.det(A) == 0:
        raise ValueError("Matrix A is singular")
    
    A_inv = np.linalg.inv(A)
    
    # Let's iterate over the sparse points
    n_iterations = 10
    u = seed_optical_flow_sparse[punto]
    u_m = u[1, 0]  # Swap the columns
    img1_gray
    I0 = img1_gray[points_selected[punto,0] - patch_half_size:points_selected[punto,0] + patch_half_size + 1,
                   points_selected[punto,1] - patch_half_size:points_selected[punto,1] + patch_half_size + 1]


    for k in range(n_iterations):
        points_interpolation = points_for_gradient[punto] + u_m 
        
        # Let's calculate x and y for the grid interpolation
        y = np.linspace(points_interpolation[0] - patch_half_size, points_interpolation[0] + patch_half_size, 2 * patch_half_size + 1)
        x = np.linspace(points_interpolation[1] - patch_half_size, points_interpolation[1] + patch_half_size, 2 * patch_half_size + 1)
        grid_mesh = np.meshgrid(y, x)
        I1 = int_bilineal(img2_gray, grid_mesh)
        I1.reshape((2 * patch_half_size + 1, 2 * patch_half_size + 1))

        # Compute the error between the patches of the two images
        error = I1 - I0
        
        # Compute the vector b from the error between patches and the gradients
        b = np.array([np.sum(error * gradients_j0[:, 1]), np.sum(error * gradients_j0[:, 0])])
        
        # Compute the update step
        u_delta = A_inv @ b
        
        # We compute the error between the patches of the two images        
        u_m += u_delta
    
    

def exercise_2_3():
    
    return None

def exercise_2_4():
    
    return None

def exercise_3():
    
    return None

### Exercices flags ###
EXERCICE_2_1 = True
EXERCICE_2_2 = True
EXERCICE_2_3 = False
EXERCICE_2_4 = False
EXERCICE_3 = False
DEBUG = True

if __name__ == "__main__":
    
    main()
    