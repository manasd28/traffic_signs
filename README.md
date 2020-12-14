# Traffic Sign Detection using Yolo-V3 #

#### Two Step Working : #####
  
  - `Detect Traffic signs into 4 broad categories using Yolo-V3.`
  
  - `Classify the Traffic sign into 43 Different Categories.`
  
### Python Dependencies and Packages : ###
  
  - `opencv-python`
  - `tensorflow`
  - `keras`
  - `numpy`

## Usage : ##
  
  ### Supports three different kinds of operation :
  
  - #### Classify the traffic signs found in an image: ####
    
    1. Add all the images you want to classify into `/test_data/images/`
    2. Run the script `/scripts/main_image.py`
    3. Find the output images with classified traffic signs in the folder `/result_data/images/`

  - #### Classify the traffic signs found in an video: ####
  
    1. Add all the images you want to classify into `/test_data/videos/`
    2. Run the script `/scripts/main_videos.py`
    3. Find the output images with classified traffic signs in the folder `/result_data/videos/`

  - #### Classify the traffic signs live using a camera: ###
    
    1. Run the script `/scripts/main_camera.py`
