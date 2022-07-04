# PCD-Cam Calibrator

This is a tool for calibrating the intrinsic matrix of a camera by manually picking points in corresonding 2D images and point clouds.

## Installation

The simplest way to install all dependencies and requirements for this project is to create a new conda environment from the supplied text file. This is done by running the following commands:

```
cd conda_envs
conda create --name pcd_cam --file pcd_cam.txt
```

## Usage

#TODO

## Roadmap

-   [x] Write 2d point picking function
-   [x] Write 3d point picking function
-   [x] Transform coordinates into correct form ( + save session perhaps )
-   [x] Write calibration function
-   [x] Write additional utils if needed
-   [ ] Write validation function
-   [ ] Perform validation for inclusion into publications
-   [ ] Update documentation with more examples, description, etc. 