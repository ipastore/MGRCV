import funciones_CV as fcv
import numpy as np
import matplotlib.pyplot as plt
import saveData

import gtFunctions as gtf

import numpy as np



if __name__ == "__main__":
    colmapSet = 9
    images_file = f'./colmap/set{colmapSet}/images.txt'  # Replace with the path to your images.txt file
    camera_poses = gtf.extract_camera_poses(images_file)

    #########################
    # Load Pose from COLMAP #
    #########################
    T_gt = {}
    T_diffReference = {}

    for camera_name in camera_poses.keys():
        R_gt = camera_poses[camera_name]['rotation_matrix']
        t_gt = camera_poses[camera_name]['translation_vector']
        T_gt[f'T_{camera_name}_gt'] = fcv.ensamble_T(R_gt, t_gt)

        R__, t__  = gtf.changePoseTransformation(R_gt, t_gt)
        T_diffReference[f'T_{camera_name}_diffReference'] = fcv.ensamble_T(R__, t__)


    #######################
    # Load 3D from COLMAP #
    #######################
    points3D_file = f'./colmap/set{colmapSet}/points3D.txt'
    points3D, pointsColor = gtf.load_points3D(points3D_file)
    print(f"Loaded {points3D.shape[0]} 3D points.")

    points3D_h = fcv.homogenizePoints(points3D.T)


    # Geometry
    T_w1 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).T

    #####################
    #  Plot 3D COLMAP   #
    #####################
    plt.figure(1)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter((T_w1@fcv.homogenizePoints(points3D.T))[0, :], (T_w1@fcv.homogenizePoints(points3D.T))[1, :], (T_w1@fcv.homogenizePoints(points3D.T))[2, :], color=pointsColor/255.0, marker='.', label='Reconstructed 3D points')
    i = 0
    for cameras in T_gt.keys():
        if cameras == 'T_IMG_0001.png_diffReference':
            fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_gt[cameras]), '-', f'C_Old')
            # fcv.drawRefSystem(ax, T_w1@ T_gt[cameras], '-.', f'C_.Old')
        else:       
            fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_gt[cameras]), '-', f'C_{i}')
            # fcv.drawRefSystem(ax, T_w1@ T_gt[cameras], '-.', f'C_.{i}')
        i+=1
    ax.legend()
    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(-5, 50, 2)
    yFakeBoundingBox = np.linspace(-5, 50, 2)
    zFakeBoundingBox = np.linspace(-5, 50, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    plt.title('COLMAP 3D points')

    #####################
    #  Plot 3D COLMAP   # ===> T <= T^-1
    #####################
    plt.figure(2)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter((T_w1@points3D_h)[0, :], (T_w1@points3D_h)[1, :], (T_w1@points3D_h)[2, :], color=pointsColor/255.0, marker='.', label='Reconstructed 3D points')
    i = 0
    for cameras in T_diffReference.keys():
        if cameras == 'T_IMG_0001.png_diffReference':
            fcv.drawRefSystem(ax, T_w1 @ T_diffReference[cameras], '-', f'C_Old')
        else:       
            fcv.drawRefSystem(ax, T_w1 @ T_diffReference[cameras], '-.', f'C_{i}')
        i+=1  
    ax.legend()
    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(-5, 50, 2)
    yFakeBoundingBox = np.linspace(-5, 50, 2)
    zFakeBoundingBox = np.linspace(-5, 50, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    plt.title('COLMAP 3D points but inverted by R and t')



    #####################
    # Load 3D from OWN  #
    #####################
    input_file = "./goodRun/output.txt"
    transformations_, points_, point_colors = saveData.load_transformations_and_points(input_file)
    T_1_own = transformations_['T_11']
    T_2_own = transformations_['T_21_best_2']
    T_3_own = transformations_['T_c3_c1_best_2']
    # T_own_old_ig = transformations_['T_old_ig']
    T_old_own = transformations_['T_old_best']

    X_wOld = points_['X_3d_old_123']
    X_123 = points_['X_3d_3_12_best']

    ######################
    #  Plot 3D Own Data  #
    ######################
    plt.figure(3)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_1_own), '-', f'C_own_1')
    fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_2_own), '-', f'C_own_2')
    fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_3_own), '-', f'C_own_3')
    fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_old_own), '-', f'C_own_old_gt')
    ax.scatter((T_w1@fcv.homogenizePoints(X_wOld.T))[0, :], (T_w1@fcv.homogenizePoints(X_wOld.T))[1, :], (T_w1@fcv.homogenizePoints(X_wOld.T))[2, :], color='g', marker='o', label='Own 3D old')
    ax.scatter((T_w1@fcv.homogenizePoints(X_123.T))[0, :], (T_w1@fcv.homogenizePoints(X_123.T))[1, :], (T_w1@fcv.homogenizePoints(X_123.T))[2, :], color='r', marker='.', label='Own 3D 123')
    xFakeBoundingBox = np.linspace(-2, 6, 2)
    yFakeBoundingBox = np.linspace(-2, 6, 2)
    zFakeBoundingBox = np.linspace(-2, 6, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    plt.title('Own 3D points')
    
    """# Scale
    scale_gt = np.linalg.norm(T_gt['T_IMG_0169.png_gt'][:3, 3])
    scale_own = np.linalg.norm((T_3_own)[:3, 3])

    # scale_factor =  scale_gt / scale_own
    scale_factor =   scale_own / scale_gt

    T_1_gt = T_gt['T_IMG_0160.png_gt']
    T_2_gt = T_gt['T_IMG_0169.png_gt']
    T_3_gt = T_gt['T_IMG_0173.png_gt']
    T_old_gt = T_gt['T_IMG_0001.png_gt']

    X_gt = fcv.homogenizePoints(points3D.T)
    X_gt *= scale_factor
    T_1_gt[:3, 3] *= scale_factor
    T_2_gt[:3, 3] *= scale_factor
    T_3_gt[:3, 3] *= scale_factor
    T_old_gt[:3, 3] *= scale_factor
    

    T_align = np.linalg.inv(T_gt['T_IMG_0160.png_gt'])

    T_1_gt = T_align @ T_1_gt
    T_2_gt = T_align @ T_2_gt
    T_3_gt = T_align @ T_3_gt
    T_old_gt = T_align @ T_old_gt

    X_gt = T_align @ X_gt """

    print(T_gt.keys())

    ###########################
    #  Align COLMAP with OWN  #
    ###########################
    # Align
    T_align = np.linalg.inv(T_gt['T_IMG_0160.png_gt'])

    # T_1_gt = T_align @ T_gt['T_IMG_0160.png_gt'] # == np.eye(4, 4)
    # T_2_gt = T_align @ T_gt['T_IMG_0169.png_gt']
    # T_3_gt = T_align @ T_gt['T_IMG_0173.png_gt']
    # T_old_gt = T_align @ T_gt['T_IMG_0001.png_gt']
    T_1_gt = T_gt['T_IMG_0160.png_gt'] @ T_align
    T_2_gt = T_gt['T_IMG_0169.png_gt'] @ T_align
    T_3_gt = T_gt['T_IMG_0173.png_gt'] @ T_align
    T_old_gt = T_gt['T_IMG_0001.png_gt'] @ T_align
    
    X_gt = np.linalg.inv(T_align) @ points3D_h

    # Scale
    scale_gt = np.linalg.norm(T_1_gt[:3, 3])
    scale_own = np.linalg.norm(T_1_own[:3, 3])
    scale_factor_1 =   scale_own / scale_gt
    print(f'Scale factor with T_1: {scale_factor_1}')

    scale_gt = np.linalg.norm(T_2_gt[:3, 3])
    scale_own = np.linalg.norm(T_2_own[:3, 3])
    scale_factor_2 =   scale_own / scale_gt
    print(f'Scale factor with T_2: {scale_factor_2}')

    scale_gt = np.linalg.norm(T_3_gt[:3, 3])
    scale_own = np.linalg.norm(T_3_own[:3, 3])
    scale_factor_3 =   scale_own / scale_gt
    print(f'Scale factor with T_3: {scale_factor_3}')

    scale_gt = np.linalg.norm(T_old_gt[:3, 3])
    scale_own = np.linalg.norm(T_old_own[:3, 3])
    scale_factor_old =   scale_own / scale_gt
    print(f'Scale factor with T_old: {scale_factor_old}')

    scale_factor = scale_factor_2

    X_gt[:3, :] *= scale_factor
    T_1_gt[:3, 3] *= scale_factor
    T_2_gt[:3, 3] *= scale_factor
    T_3_gt[:3, 3] *= scale_factor
    T_old_gt[:3, 3] *= scale_factor

    plt.figure(4)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter((T_w1@X_gt)[0, :], (T_w1@X_gt)[1, :], (T_w1@X_gt)[2, :], color='c', marker='.', label='Own 3D colmap')
    X_wOld = points_['X_3d_old_123']
    ax.scatter((T_w1@fcv.homogenizePoints(X_wOld.T))[0, :], (T_w1@fcv.homogenizePoints(X_wOld.T))[1, :], (T_w1@fcv.homogenizePoints(X_wOld.T))[2, :], color='g', marker='o', label='Own 3D old')
    X_123 = points_['X_3d_3_12_best']
    ax.scatter((T_w1@fcv.homogenizePoints(X_123.T))[0, :], (T_w1@fcv.homogenizePoints(X_123.T))[1, :], (T_w1@fcv.homogenizePoints(X_123.T))[2, :], color='r', marker='.', label='Own 3D 123')
    # fcv.plotNumbered3DPoints(ax, (T_w1@fcv.homogenizePoints(X_123.T)), 'r')
    # fcv.plotNumbered3DPoints(ax, (T_w1@X_gt), 'g')
    fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_1_gt), '-.', f'C_1_gt')
    fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_2_gt), '-.', f'C_2_gt')
    fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_3_gt), '-.', f'C_3_gt')
    fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_old_gt), '-.', f'C_old_gt')

    fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_1_own), '-', f'C_1_own')
    fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_2_own), '-', f'C_2_own')
    fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_3_own), '-', f'C_3_own')
    fcv.drawRefSystem(ax, T_w1@np.linalg.inv(T_old_own), '-', f'C_old_own')
    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(-2, 6, 2)
    yFakeBoundingBox = np.linspace(-2, 6, 2)
    zFakeBoundingBox = np.linspace(-2, 6, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')


    plt.legend()
    plt.title('3D comparison')


    #################
    # Load dense 3D #
    #################
    points3D_file_dense = f'./colmap/set10/dense/points3D.txt'
    points3D_dense, pointsColor_dense = gtf.load_points3D(points3D_file_dense)
    print(f"Loaded {points3D_dense.shape[0]} 3D points.")

    points3D_h_dense = fcv.homogenizePoints(points3D_dense.T)
    X_dense = np.linalg.inv(T_align) @ points3D_h_dense
    X_dense[:3, :] *= scale_factor
    # ax.scatter((T_w1@X_dense)[0, :], (T_w1@X_dense)[1, :], (T_w1@X_dense)[2, :], color=pointsColor_dense/255.0, marker='.', label='Reconstructed 3D points')


    images_file_27 = f'./colmap/set10/images.txt'  # Replace with the path to your images.txt file
    camera_poses_27 = gtf.extract_camera_poses(images_file_27)

    T_27_dict = {}
    
    for camera_name in camera_poses_27.keys():
            R_gt = camera_poses_27[camera_name]['rotation_matrix']
            t_gt = camera_poses_27[camera_name]['translation_vector']
            T_27_dict[f'T_{camera_name}_gt'] = fcv.ensamble_T(R_gt, t_gt)
            print(f'{camera_name} processed')

    T_align_27 =  np.linalg.inv(T_27_dict['T_IMG_0160.png_gt'])

    scale_factor_27 = np.linalg.norm(T_2_own[:3, 3]) / np.linalg.norm(T_27_dict['T_IMG_0169.png_gt'][:3, 3])
    
    T_27_aligned ={}
    for poses in T_27_dict.keys():
        T_27_aligned[poses] = T_27_dict[poses] @ T_align_27
        T_27_aligned[poses][:3, 3] *= scale_factor_27
        print(f'{poses} after alignment: {T_27_aligned[poses]}')

    ####################
    # 3D point matches #
    ####################
    points3D_file27 = f'./colmap/set10/dense/points3D.txt'
    points3D_27, pointsColor_27 = gtf.load_points3D(points3D_file27)
    print(f"Loaded {points3D_27.shape[0]} 3D points.")

    points3D_27_h = fcv.homogenizePoints(points3D_27.T)
    
    X_colmap27_aligned = np.linalg.inv(T_align_27) @ points3D_27_h
    X_colmap27_aligned[:3, :] *= scale_factor_27

    # #####################
    # #  Plot 3D matches  #
    # #####################
    # plt.figure(100)
    # ax = plt.axes(projection='3d', adjustable='box')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # ax.scatter((T_w1 @ X_colmap27_aligned)[0, :], (T_w1 @ X_colmap27_aligned)[1, :], (T_w1 @ X_colmap27_aligned)[2, :], color=pointsColor_27/255.0, marker='.', label='3D cloud from colmap with 27 images')
    # ax.scatter((T_w1@X_dense)[0, :], (T_w1@X_dense)[1, :], (T_w1@X_dense)[2, :], color='g', marker='.', label='Reconstructed 3D points')
    # for poses in T_27_aligned.keys():
    #     if poses == 'T_IMG_0001.png_gt':
    #         fcv.drawRefSystem(ax, T_w1 @ np.linalg.inv(T_27_aligned[poses]), '-', 'Old')
    #     else:
    #         fcv.drawRefSystem(ax, T_w1 @ np.linalg.inv(T_27_aligned[poses]), '-.', '')

    # fcv.drawRefSystem(ax, T_w1 @ T_1_own, '--', 'gt_1')
    # fcv.drawRefSystem(ax, T_w1 @ np.linalg.inv(T_2_own), '--', 'gt_c2')
    # fcv.drawRefSystem(ax, T_w1 @ np.linalg.inv(T_3_own), '--', 'gt_c3')
    # fcv.drawRefSystem(ax, T_w1 @ np.linalg.inv(T_old_own), '--', 'gt_cOld')

    # #Matplotlib does not correctly manage the axis('equal')
    # xFakeBoundingBox = np.linspace(-5, 15, 2)
    # yFakeBoundingBox = np.linspace(-1, 19, 2)
    # zFakeBoundingBox = np.linspace(-10, 10, 2)
    # plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    # plt.title('All 27 images')


    #############################
    #  Plot DENSE with 27 IMGS  #
    #############################
    plt.figure(101)
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter((T_w1@X_dense)[0, :50000], (T_w1@X_dense)[1, :50000], (T_w1@X_dense)[2, :50000], color='b', marker=1, label='Reconstructed 3D points')
    ax.scatter((T_w1@X_dense[:, 16822])[0], (T_w1@X_dense[:, 16822])[1], (T_w1@X_dense[:, 16822])[2], color='r', marker='x', label='Reconstructed 3D points')
    ax.scatter((T_w1@X_dense[:, 74566])[0], (T_w1@X_dense[:, 74566])[1], (T_w1@X_dense[:, 74566])[2], color='r', marker='x', label='Reconstructed 3D points')

    # fcv.plotNumbered3DPoints(ax, (T_w1@fcv.homogenizePoints(X_dense.T).T), 'r')
    # for poses in T_27_aligned.keys():
    #     if poses == 'T_IMG_0001.png_gt':
    #         fcv.drawRefSystem(ax, T_w1 @ np.linalg.inv(T_27_aligned[poses]), '-', 'Old')
    #     else:
    #         fcv.drawRefSystem(ax, T_w1 @ np.linalg.inv(T_27_aligned[poses]), '-.', '')

    fcv.drawRefSystem(ax, T_w1 @ T_1_own, '--', 'gt_1')
    fcv.drawRefSystem(ax, T_w1 @ np.linalg.inv(T_2_own), '--', 'gt_c2')
    fcv.drawRefSystem(ax, T_w1 @ np.linalg.inv(T_3_own), '--', 'gt_c3')
    fcv.drawRefSystem(ax, T_w1 @ np.linalg.inv(T_old_own), '--', 'gt_cOld')

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(-5, 15, 2)
    yFakeBoundingBox = np.linspace(-1, 19, 2)
    zFakeBoundingBox = np.linspace(-10, 10, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    plt.title('Fancy')













    scaleRW = 120.0

    # COLMAP
    X_tower1_colmap=points3D_h[:, 7]
    X_tower2_colmap=points3D_h[:, 35]
    scale_colmap = np.linalg.norm(X_tower1_colmap - X_tower2_colmap)
    sfRW_colmap = scaleRW / scale_colmap 

    X_RW_colmap = np.copy(points3D_h)

    X_RW_colmap = X_RW_colmap

    X_RW_colmap[:3, :] *= sfRW_colmap

    T_1_gt_RW = np.copy(T_gt['T_IMG_0160.png_gt'])
    T_2_gt_RW = np.copy(T_gt['T_IMG_0169.png_gt'])
    T_3_gt_RW = np.copy(T_gt['T_IMG_0173.png_gt'])
    T_Old_gt_RW = np.copy(T_gt['T_IMG_0001.png_gt'])

    T_1_gt_RW[:3, 3] *= sfRW_colmap
    T_2_gt_RW[:3, 3] *= sfRW_colmap
    T_3_gt_RW[:3, 3] *= sfRW_colmap
    T_Old_gt_RW[:3, 3] *= sfRW_colmap

    #OWN

    X_tower1_own=(T_w1@fcv.homogenizePoints(X_123.T))[:, 7]
    X_tower2_own=(T_w1@fcv.homogenizePoints(X_123.T))[:, 34]
    scale_own = np.linalg.norm(X_tower1_own - X_tower2_own)


    sfRW_own = scaleRW / scale_own

    X_RW_own = np.copy(fcv.homogenizePoints(X_123.T))

    X_RW_own[:3, :] *= sfRW_own


    T_1_own_RW = np.copy(T_1_own)
    T_2_own_RW = np.copy(T_2_own)
    T_3_own_RW = np.copy(T_3_own)
    T_Old_own_RW = np.copy(T_old_gt)
    T_1_own_RW[:3, 3] *= sfRW_colmap
    T_2_own_RW[:3, 3] *= sfRW_colmap
    T_3_own_RW[:3, 3] *= sfRW_colmap
    T_Old_own_RW[:3, 3] *= sfRW_colmap

    plt.figure(40)
    fig = plt.gcf()
    fig.patch.set_alpha(0) 
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_facecolor('none') 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # ax.scatter((T_w1@X_RW_own)[0, :], (T_w1@X_RW_own)[1, :], (T_w1@X_RW_own)[2, :], color='b', marker='.', label='Own 3D old')
    ax.scatter((T_w1@X_RW_colmap)[0, :], (T_w1@X_RW_colmap)[1, :], (T_w1@X_RW_colmap)[2, :], color='c', marker='.', label='Own 3D colmap')
    # fcv.plotNumbered3DPoints(ax, (T_w1@fcv.homogenizePoints(X_123.T)), 'r')
    # fcv.drawRefSystem_RW(ax, T_w1@np.linalg.inv(T_1_own_RW), '-.', f'C_1_own', scale=100.0, lWidth=3.0)
    # fcv.drawRefSystem_RW(ax, T_w1@np.linalg.inv(T_2_own_RW), '-.', f'C_2_own', scale=100.0, lWidth=3.0)
    # fcv.drawRefSystem_RW(ax, T_w1@np.linalg.inv(T_3_own_RW), '-.', f'C_3_own', scale=100.0, lWidth=3.0)
    # fcv.drawRefSystem_RW(ax, T_w1@np.linalg.inv(T_Old_own_RW), '-.', f'C_old_own', scale=100.0, lWidth=3.0)
    fcv.drawRefSystem_RW(ax, T_w1@np.linalg.inv(T_1_gt_RW), ':', f'C_1_gt', scale=100.0, lWidth=2.0)
    fcv.drawRefSystem_RW(ax, T_w1@np.linalg.inv(T_2_gt_RW), ':', f'C_2_gt', scale=100.0, lWidth=2.0)
    fcv.drawRefSystem_RW(ax, T_w1@np.linalg.inv(T_3_gt_RW), ':', f'C_3_gt', scale=100.0, lWidth=2.0)
    fcv.drawRefSystem_RW(ax, T_w1@np.linalg.inv(T_Old_gt_RW), ':', f'C_old_gt', scale=100.0, lWidth=2.0)
    ax.legend()
    plt.title('Initial set of 3D')
    ax.view_init(elev=90, azim=0)

    #Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(-1000, 1000, 2)
    yFakeBoundingBox = np.linspace(-1000, 1000, 2)
    zFakeBoundingBox = np.linspace(-1000, 1000, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    # plt.savefig("plot_with_transparent_background.png", dpi=3000, transparent=True)
    # Focus on a specific region by setting axis limits
    ax.set_xlim(-100, 400)  # Adjust X-axis limits (e.g., focus on 0 to 200)
    ax.set_ylim(-50, 450)  # Adjust Y-axis limits (e.g., focus on 0 to 300)
    ax.set_zlim(-200, 300)  # Adjust Z-axis limits (e.g., focus on 0 to 500)

    plt.show()

    dummy = []




