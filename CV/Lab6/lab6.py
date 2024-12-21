from utils.matrixOperationsCV import *
from utils.NCCTemplate import *
from utils.interpolationFunctionsCV import *
from utils.plotGroundTruthOpticalFlow import *



def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def make_swapped_patch(point, patch_half_size):
   
    """ 
    Creates a patch of coordinates centered around a given point, then swaps the x and y coordinates.
    
    :param point: (x, y) representing the center point of the patch.
    :param patch_half_size (int): The half size of the patch. The full patch size will be (2 * patch_half_size + 1).
    :return: np.ndarray (float): A 2D array of shape ((2 * patch_half_size + 1) ** 2, 2) containing the swapped coordinates.
    """
    patch = np.zeros((patch_half_size*2+1, patch_half_size*2+1, 2), dtype=float)
    for j in range(0, patch_half_size*2+1):
        for k in range(0, patch_half_size*2+1):
            patch[j, k, 0] = point[0] - patch_half_size + j
            patch[j, k, 1] = point[1] - patch_half_size + k
        
    patch = patch.reshape(-1, 2)
    patch_swapped = [[point_y, point_x] for point_x, point_y in patch]
    patch_swapped = np.array(patch_swapped, dtype=float)
    
    return patch_swapped
        
def main():
    
    # We load the images
    img1 = read_image("./data/frame10.png")
    img2 = read_image("./data/frame11.png")
    img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    # List of sparse points selected
    points_selected = np.loadtxt('./data/points_selected.txt')
    points_selected = points_selected.astype(int)
    
    ### EXERCICE 2.1: Kannala-Brandt Model ###
    
    if EXERCICE_2_1:
        seed_optical_flow_sparse = exercise_2_1(img1_gray,img2_gray,points_selected)
    
    if EXERCICE_2_2:
        optical_flow_subpixel = exercise_2_2(img1_gray,img2_gray,points_selected,seed_optical_flow_sparse)
        
    if EXERCICE_2_3:
        exercise_2_3(img1,points_selected,seed_optical_flow_sparse,optical_flow_subpixel)
        

                     
def exercise_2_1(img1_gray,img2_gray,points_selected):
    
    print("\n*************************************************************************************")
    print("* EXERCICE 2.1: Motion  by using Normalized Cross Correlation(NCC) brute-force search *")
    print("*************************************************************************************\n")
    


    # Define the template size and the searching area size
    template_size_half = 5          # Define a template of 5x5 pixels. This patch will move around the searching area defined by the searching_area_size
    searching_area_size: int = 15   # Define a region of 15x15 pixels around the pixel to search for the best match using the template patch

    seed_optical_flow_sparse = np.zeros((points_selected.shape))
    for k in range(0,points_selected.shape[0]):
        i_flow, j_flow = seed_estimation_NCC_single_point(img1_gray, img2_gray, points_selected[k,1], points_selected[k,0], template_size_half, searching_area_size)
        seed_optical_flow_sparse[k,:] = np.hstack((j_flow,i_flow))

    print(f"Seed optical flow: {seed_optical_flow_sparse}")
    
    return seed_optical_flow_sparse

def exercise_2_2(img1_gray,img2_gray,points_selected,seed_optical_flow_sparse):
    
    print("\n*******************************************************************************")
    print("* EXERCICE 2.2 and 2.3: Lucas Kanade approach that refines your NCC brute-force search *")
    print("*******************************************************************************\n")
    

    # Define the template size and the searching area size
    patch_half_size = 5          # Define a template of 5x5 pixels. This patch will move around the searching area defined by the searching_area_size
    epsilon = 0.0001            # Define the epsilon value for the convergence of the algorithm

    # Initialize the optical flow
    optical_flow_subpixel= np.zeros(seed_optical_flow_sparse.shape, dtype=float)

    # Iterate over the selected points
    for i in range(points_selected.shape[0]):
        point = points_selected[i]
        debug_print(f"Point: {point}")

        # Make patch of coordinates to calculate gradient in image 1
        patch_for_gradient = make_swapped_patch(point, patch_half_size)

        # Compute the gradients of the image [y,x]
        gradients = numerical_gradient(img1_gray, patch_for_gradient)
        
        # Compute the matrix elements of A
        A_11 = np.sum(gradients[:, 0] ** 2)
        A_12 = np.sum(gradients[:, 0] * gradients[:, 1])
        A_22 = np.sum(gradients[:, 1] ** 2)
        A = np.array([[A_11, A_12], [A_12, A_22]])
        
        # Check A is invertible
        assert np.linalg.det(A) != 0
        
        # Check if the matrix is nearly singular (similar as previous)
        if np.linalg.det(A) < 1e-6:  # Example threshold
            print("Matrix A is nearly singular, should skip update.")
            continue
        
        # Compute the inverse of A
        A_inv = np.linalg.inv(A)
        
        # copy seed_optical_flow_sparse for sanitization
        seed_optical_flow_sparse_copy = seed_optical_flow_sparse.copy()
        u = seed_optical_flow_sparse_copy[i]
        delta_u = np.array([1, 1])

        # Make a patch (I0) with the selected point as the center of image 1
        I0 = np.zeros((patch_half_size*2+1, patch_half_size*2+1), dtype=float)
        for j in range(0, patch_half_size*2+1):
            for k in range(0, patch_half_size*2+1):
                I0[j, k] = img1_gray[point[1] - patch_half_size + k, point[0] - patch_half_size + j]

        # Ravel to compare with I1 in the loop
        I0 = I0.ravel()

        # Initialize parameters for the iteration
        k = 0
        point1 = point + u

        while (np.sqrt(np.sum(delta_u ** 2))) >= epsilon:
            debug_print(f"#################iteration {k}##################")
            debug_print("point1: ", point1, sep="\n")

            # Make patch of coordinates to interpolate in image 2
            patch_for_interpolation = make_swapped_patch(point1, patch_half_size)

            # Compute the interpolated patch in the second image
            I1 = int_bilineal(img2_gray, patch_for_interpolation)

            # Compute the error between the patches of the two images
            error = I1 - I0

            # Compute the vector b from the error between patches and the gradients
            b = np.array([np.sum(error * gradients[:, 0]), np.sum(error * gradients[:, 1])])
            b = -b

            # Compute the update step
            delta_u = A_inv @ b
            
            # Updates for inner iterations
            u += delta_u
            k += 1
            point1 = point + u
            debug_print(f"delta_u:\n {delta_u},\n u:\n {u}, iteration {k-1} ")
        
        #Update the optical flow
        optical_flow_subpixel[i] = u

    print(f"Finished u: {optical_flow_subpixel}")
    
    return optical_flow_subpixel
        
def exercise_2_3(img1,points_selected,seed_optical_flow_sparse,optical_flow_subpixel):
    print("\n*******************************************************************************")
    print("* EXERCICE 2.4: PLotting solutions *")
    print("*******************************************************************************\n")

    # Load the ground truth optical flow
    flow_12 = read_flo_file("./data/flow10.flo", verbose=True)
    flow_gt = flow_12[points_selected[:, 1].astype(int), points_selected[:, 0].astype(int)].astype(float)
    
    ## Sparse optical flow
    flow_est_sparse = seed_optical_flow_sparse
    flow_est_sparse_norm = np.sqrt(np.sum(flow_est_sparse ** 2, axis=1))
    error_sparse = flow_est_sparse - flow_gt
    error_sparse_norm = np.sqrt(np.sum(error_sparse ** 2, axis=1))

    ## Subpixel optical flow
    flow_est_subpixel = optical_flow_subpixel
    flow_est_subpixel_norm = np.sqrt(np.sum(flow_est_subpixel ** 2, axis=1))
    error_subpixel = flow_est_subpixel - flow_gt
    error_subpixel_norm = np.sqrt(np.sum(error_subpixel ** 2, axis=1))

    # Plot results for sparse optical flow
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].imshow(img1)
    axs[0, 0].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[0, 0].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(flow_est_sparse_norm[k]), color='r')
    axs[0, 0].quiver(points_selected[:, 0], points_selected[:, 1], flow_est_sparse[:, 0], flow_est_sparse[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[0, 0].title.set_text('Sparse Optical flow')
    
    axs[0, 1].imshow(img1)
    axs[0, 1].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[0, 1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_sparse_norm[k]), color='r')
    axs[0, 1].quiver(points_selected[:, 0], points_selected[:, 1], error_sparse[:, 0], error_sparse[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[0, 1].title.set_text('Sparse Error with respect to GT')
    
    axs[1, 0].imshow(img1)
    axs[1, 0].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[1, 0].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(flow_est_subpixel_norm[k]), color='r')
    axs[1, 0].quiver(points_selected[:, 0], points_selected[:, 1], flow_est_subpixel[:, 0], flow_est_subpixel[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[1, 0].title.set_text('Subpixel Optical flow')
    
    axs[1, 1].imshow(img1)
    axs[1, 1].plot(points_selected[:, 0], points_selected[:, 1], '+r', markersize=15)
    for k in range(points_selected.shape[0]):
        axs[1, 1].text(points_selected[k, 0] + 5, points_selected[k, 1] + 5, '{:.2f}'.format(error_subpixel_norm[k]), color='r')
    axs[1, 1].quiver(points_selected[:, 0], points_selected[:, 1], error_subpixel[:, 0], error_subpixel[:, 1], color='b', angles='xy', scale_units='xy', scale=0.05)
    axs[1, 1].title.set_text('Subpixel Error with respect to GT')
    
    plt.tight_layout()
    plt.show()

    return None


### Exercices flags ###
EXERCICE_2_1 = True
EXERCICE_2_2 = True
EXERCICE_2_3 = True
DEBUG = False

if __name__ == "__main__":
    
    main()
    