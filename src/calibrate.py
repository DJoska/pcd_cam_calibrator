import cv2 as cv
import numpy as np
import os
import open3d

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
    import open3d
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

def calibrate_intrisics(img_pts, obj_pts, im_folder, pcd_folder, image_type=".png"):
    """
    Makes use of OpenCV's camera calibration functions to calibrate based on picked points
    """

    # get lists of absolute image and pcd filepaths
    images = [os.path.join(im_folder, im) for im in os.listdir(im_folder) if image_type in im]
    print(images)
    pcds = [os.path.join(pcd_folder, pcd) for pcd in os.listdir(pcd_folder) if ".pcd" in pcd]
    print(pcds)

    all_obj_pts = []
    all_img_pts = []
    for im in images:
        img_mat = pick_2d_points(im)
        all_img_pts.append(img_mat)
    
    for pcd in pcds:
        obj_mat = pick_3d_points(pcd)
        all_obj_pts.append(obj_mat)
    
    print("Image Points:")
    print(all_img_pts)
    print("Object Points:")
    print(all_obj_pts)


def get_reprojection_errors(plot=False):
    """
    Returns reprojection errors for a given calibration session
    TODO
    """
    return 1

if __name__=="__main__":
    example_png = "/Users/user/Desktop/wildpose_work/pngs/258.999313280.png"
    example_pcd = "/Users/user/Desktop/wildpose_work/pcds/258.999313280.pcd"
    pts2d = pick_2d_points(example_png)
    pts3d = pick_3d_points(example_pcd)

    print(pts2d)
    print(pts3d)
