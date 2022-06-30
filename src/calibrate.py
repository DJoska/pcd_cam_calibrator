import cv2 as cv
import numpy as np
import os
import open3d
import re

def natural_sort(l):
    """
    Natural sort function for filenames
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def pick_2d_points(image_path):
    """
    Opens a GUI to pick 2d coordinates in an image file
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
    """
    a = open3d.io.read_point_cloud(pcd_path)

    vis = open3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(a)
    vis.run()
    vis.destroy_window()

    print(vis.get_picked_points())
    obj_pts = []
    for i in vis.get_picked_points():
        print(a.points[i])
        obj_pts.append(a.points[i])
    
    return obj_pts

def calibrate_intrisics(im_folder, pcd_folder, image_type=".png"):
    """
    Makes use of OpenCV's camera calibration functions to calibrate based on picked points
    """

    # get lists of absolute image and pcd filepaths
    images = natural_sort([os.path.join(im_folder, im) for im in os.listdir(im_folder) if image_type in im])
    print(images)
    pcds = natural_sort([os.path.join(pcd_folder, pcd) for pcd in os.listdir(pcd_folder) if ".pcd" in pcd])
    print(pcds)

    all_obj_pts = []
    all_img_pts = []
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
        [700, 0, 426],
        [0, 700, 240],
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
    if ret:
        return k, d, r, t
    else:
        return 0,0,0,0


def get_reprojection_errors(plot=False):
    """
    Returns reprojection errors for a given calibration session
    TODO
    """
    return 1

if __name__=="__main__":
    example_png = "/Users/user/Desktop/wildpose_work/pngs/258.999313280.png"
    example_pcd = "/Users/user/Desktop/wildpose_work/pcds/258.999313280.pcd"
    png_folder = "/Users/user/Desktop/pngs"
    pcd_folder = "/Users/user/Desktop/pcds"
    #pts2d = pick_2d_points(example_png)
    #pts3d = pick_3d_points(example_pcd)
    #print(pts2d)
    #print(pts3d)
    calibrate_intrisics(png_folder, pcd_folder)
