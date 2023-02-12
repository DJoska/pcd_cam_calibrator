import cv2 as cv
import numpy as np
import os
import open3d
import pickle
from utils import natural_sort, load_pickle
import matplotlib.pyplot as plt

def pick_2d_points(image_path):
    """
    Opens a GUI to pick 2d coordinates in an image file

    Returns:
    img_points: 3-D array
                An array of the image points extracted
    ----
    Parameters
    ----
    image_path: string
                The absolute file path of an image to be picked from
    """    
    def click_event(event, x, y, flags, params):
        """
        Callback function for left moust button clicks on the image
        """
        if event == cv.EVENT_LBUTTONDOWN:
    
            print([x,y])
            cv.drawMarker(img, (x, y),(0,0,255), markerType=cv.MARKER_CROSS, 
                markerSize=40, thickness=2, line_type=cv.LINE_AA)
            cv.imshow("image", img)
            img_pts.append([x,y])
            
    img = cv.imread(image_path, 1)
    img_pts = []
 
    cv.imshow('image', img)
    cv.setMouseCallback('image', click_event, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return img_pts

def pick_3d_points(pcd_path):
    """
    Opens a GUI to pick 3d coordinates in a point cloud file

    Returns:
    obj_points: 3-D array
                An array of the object points extracted
    ----
    Parameters
    ----
    pcd_path: string
                The absolute file path of a pcd to be picked from

    """
    a = open3d.io.read_point_cloud(pcd_path)
    #print(a.points["rgb"].values)

    vis = open3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(a)
    vis.run()
    vis.destroy_window()

    print(vis.get_picked_points())
    obj_pts = []
    for i in vis.get_picked_points():
        #print(a.points[i])
        point = [1000*a.points[i][1], 1000*a.points[i][2], 1000*a.points[i][0]]
        print(point)
        obj_pts.append(point)
    
    return obj_pts

def calibrate_intrisics(im_folder, pcd_folder, image_type=".png"):
    """
    Makes use of OpenCV's camera calibration functions to calibrate based on picked points

    Returns:
    K:      2-D array
            The calculated camera intrinsic matrix
    d:      array
            The distortion coefficients
    r:      array
            The rotation vectors
    t:      array
            The translation vectors
    ----
    Parameters
    ----
    im_folder:  string
                The absolute folder path of images to be picked from
    pcd_folder: string
                The absolute folder path of pcds to be picked from
    image_type: string
                The image file extension - png by default
    
    """

    # get lists of absolute image and pcd filepaths
    images = natural_sort([os.path.join(im_folder, im) for im in os.listdir(im_folder) if image_type in im])
    print(images)
    pcds = natural_sort([os.path.join(pcd_folder, pcd) for pcd in os.listdir(pcd_folder) if ".pcd" in pcd])
    print(pcds)

    all_img_pts = []
    all_obj_pts = []

    for im in images:
        img_mat = pick_2d_points(im)
        all_img_pts.append(img_mat)
    
    for pcd in pcds:
        obj_mat = pick_3d_points(pcd)
        all_obj_pts.append(obj_mat)

    first_im = cv.imread(images[0])
    first_im_gray = cv.cvtColor(first_im, cv.COLOR_BGR2GRAY)

    all_img_pts_m = np.asarray(all_img_pts, dtype=np.float32)
    all_obj_pts_m = np.asarray(all_obj_pts, dtype=np.float32)

    print("Image Points:")
    print(all_img_pts_m)
    print("Object Points:")
    print(all_obj_pts_m)

    intrinsic_guess = np.asarray([
        [5000, 0, 600],
        [0, 5000, 300],
        [0, 0, 1]
    ], dtype=np.float32)

    ret, k, d, r, t = cv.calibrateCamera(all_obj_pts_m, all_img_pts_m, first_im_gray.shape[::-1], intrinsic_guess, None, flags=cv.CALIB_USE_INTRINSIC_GUESS)
    print("Calibration matrix")
    print(k)
    print("Distortion Coeffs")
    print(d)
    print("R vecs")
    print(r)
    print("t vecs")
    print(t)

    points_dict = {"image": all_img_pts_m, "object": all_obj_pts_m}
    
    with open(r"./results/calib.pickle", "wb") as output_file:
        pickle.dump((k, d, r, t, ret), output_file)

    with open(r"./results/points.pickle", "wb") as output_file:
        pickle.dump((points_dict), output_file)

    return k, d, r, t

def reproject_world_points(pcd_path, image_path, cam_matrix, d, image_type=".png"):
    """
    Reprojects the 3D points from a given pcd file to the image plane

    Returns:
    ----
    None
    ----
    Parameters:
    ----
    pcd_path:   string
                The absolute path of the folder of pcds to reproject
    image_path: strin
                The absolute path of the folder of images to reproject
    cam_matrix: 3x3 Numpy array
                The calculated intrinsic matrix of the camera
    d:          1x5 Numpy array
                The calculated distortion coefficients of the camera
    """
    images = natural_sort([os.path.join(image_path, im) for im in os.listdir(image_path) if image_type in im])
    print(images)
    pcds = natural_sort([os.path.join(pcd_path, pcd) for pcd in os.listdir(pcd_path) if ".pcd" in pcd])
    print(pcds)

    all_obj_pts=[]

    for pcd in pcds:
        obj_mat = pick_3d_points(pcd)
        all_obj_pts.append(obj_mat)

        all_obj_pts_m = np.asarray(all_obj_pts, dtype=np.float32)

        print("Object Points:")
        print(all_obj_pts_m)

        rvec = tvec = (0,0,0)

        cam_matrix_m = np.asarray(cam_matrix)
        d_m = np.asarray(d)

        img_pts = cv.projectPoints(all_obj_pts_m, rvec, tvec, cam_matrix_m, d_m)
        print(img_pts[0])

        image_fp = images[0]
        im = cv.imread(image_fp, 1)
        im_inverted = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        plt.imshow(im_inverted)
        for pt in img_pts[0]:
            print(pt)
            plt.scatter(pt[0][0], pt[0][1])
        plt.show()

    return(0)

def reproject_points_file(calib_filepath = "./results/calib.pickle", points_filepath = "./results/points.pickle"):
    """
    Calculates reprojection errors from points file
    """
    mat = load_pickle(calib_filepath)
    points = load_pickle(points_filepath)
    img_pts = points["image"]
    obj_pts = points["object"]

    k = mat[0]
    err_arr = []
    flat_error_list = []

    for i,obj in enumerate(obj_pts):
        rvec = tvec = (0,0,0)
        img_pts_projected = cv.projectPoints(obj, rvec, tvec, k, mat[1])[0]
        img_pts_actual = img_pts[i]
        print("Projected:")
        print(img_pts_projected)
        print("Actual")
        print(img_pts_actual)
        tot = 0
        for j in range(len(img_pts_actual)):
            signed_dist_x = (img_pts_projected[j][0][0]-img_pts_actual[j][0])
            signed_dist_y = (img_pts_projected[j][0][1]-img_pts_actual[j][1])
            norm_dist = np.sqrt( abs((img_pts_projected[j][0][0]-img_pts_actual[j][0]) * (img_pts_projected[j][0][1]-img_pts_actual[j][1]) ))
            print(norm_dist)
            tot+=norm_dist
            flat_error_list.append(signed_dist_x)
            flat_error_list.append(signed_dist_y)
        
        err = tot/(len(img_pts_actual))
        print(err)
        err_arr.append(err)

    print(err_arr)
    print(np.mean(err_arr))
    print(np.std(err_arr))

    return(flat_error_list)
    
def plot_histograms():
    """
    Plots histograms for reprojection errors
    """

    nonplanar_5m = reproject_points_file(calib_filepath="/Users/user/Desktop/nonplanar/5m/calib.pickle", 
        points_filepath="/Users/user/Desktop/nonplanar/5m/points.pickle")
    planar_5m = reproject_points_file(calib_filepath="/Users/user/Desktop/planar/5m/calib.pickle", 
        points_filepath="/Users/user/Desktop/planar/5m/points.pickle")    
    
    nonplanar_50m = reproject_points_file(calib_filepath="/Users/user/Desktop/nonplanar/50m/calib.pickle", 
        points_filepath="/Users/user/Desktop/nonplanar/50m/points.pickle")
    planar_50m = reproject_points_file(calib_filepath="/Users/user/Desktop/planar/50m/calib.pickle", 
        points_filepath="/Users/user/Desktop/planar/50m/points.pickle")    

    nonplanar_100m = reproject_points_file(calib_filepath="/Users/user/Desktop/nonplanar/100m/calib.pickle", 
        points_filepath="/Users/user/Desktop/nonplanar/100m/points.pickle")
    planar_100m = reproject_points_file(calib_filepath="/Users/user/Desktop/planar/100m/calib.pickle", 
        points_filepath="/Users/user/Desktop/planar/100m/points.pickle")    
    
    plt.subplot(3,2,1)
    plt.hist(planar_5m, bins=20)
    plt.title("Planar rig at 5m, n=120")
    plt.xlabel("Error in px")
    plt.ylabel("No. of errors")

    plt.subplot(3,2,2)
    plt.hist(nonplanar_5m, bins=20)
    plt.title("Non-planar rig at 5m, n=120")
    plt.xlabel("Error in px")
    plt.ylabel("No. of errors")

    plt.subplot(3,2,3)
    plt.hist(planar_50m, bins=20)
    plt.title("Planar rig at 50m, n=120")
    plt.xlabel("Error in px")
    plt.ylabel("No. of errors")

    plt.subplot(3,2,4)
    plt.hist(nonplanar_50m, bins=20)
    plt.title("Non-planar rig at 50m, n=120")
    plt.xlabel("Error in px")
    plt.ylabel("No. of errors")

    plt.subplot(3,2,5)
    plt.hist(planar_100m, bins=20)
    plt.title("Planar rig at 100m, n=120")
    plt.xlabel("Error in px")
    plt.ylabel("No. of errors")

    plt.subplot(3,2,6)
    plt.hist(nonplanar_100m, bins=20)
    plt.title("Non-planar rig at 100m, n=120")
    plt.xlabel("Error in px")
    plt.ylabel("No. of errors")

    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    plt.show()
    

if __name__ == "__main__":
    im_folder = "/Users/user/Desktop/wildpose_thesis/planar/5m/figims"
    pcd_folder = "/Users/user/Desktop/wildpose_thesis/planar/5m/figpcd"

    # STEP 1 - Labelling and calibration
    # calibrate_intrisics(im_folder, pcd_folder, image_type = ".png")

    # STEP 2 - Inspection of results
    #reproject_points_file()

    # STEP 3 - Plotting
    
    #reproject_points_file(calib_filepath="/Users/user/Desktop/nonplanar/5m/calib.pickle", points_filepath="/Users/user/Desktop/nonplanar/5m/points.pickle")
    
    #plot_histograms()