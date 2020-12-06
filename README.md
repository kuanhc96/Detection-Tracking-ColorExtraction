Files that I wrote:
./Base_Evaluator.py: Evaluator class for object detection. Used to evaluate object detectors on the WILDTRACK dataset(https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)

./Yolov5_Evaluator.py: inherits Base_Evaluator, used to evaluate the yolov5 detector on the WILDTRACK dataset. Contains
convenience method for accessing bbox information

./ColorLabeler.py: implements the ColorLabeler class. Used to label RGB colors with reader-friendly labels

./ImageProcessing.py: used to process images for better color detection results

./video_to_frames.py: converts a video (.mp4) into a series of frames, labeled in a way that is ascii sortable
(00000000.jpg to 99999999.jpg) for the tracker to process

./yolov5_deepsort_db.py: code adopted/modified from
(https://github.com/ZQPei/deep_sort_pytorch/blob/master/yolov3_deepsort.py). 
The original author implemented the code using the yolov3 detector.
In the run() method, I add processes that access information using the yolov5 detector, process images for my purposes,
and stores the information in a database

./db_querier.py: used to access the results stored in bbox_with_colors.db. Try running: python db_querier.py --id 10
./create_color_db.py: used to convert bbox_wirth_colors.db to database with rgb-labeled information


Files that I did not write:
./edge_model.yml: used to implement the edge detection step

./models: contains yolov5 neural network models
./utils: utility functions for the yolov5 detector

./deep_sort: DeepSORT algorithm implemented by ZQPei (https://github.com/ZQPei/deep_sort_pytorch)
./tracking_configs: contains model for DeepSORT
./tracking_utils: utility functions for the tracker, such as drawing bboxes, ID's, logging info, and warnings

Other files came with the original repo and are irrelavant for this project

To get the code running, required libraries need to be installed via pip or conda. These specifications can be found in
requirements.txt and tracking_requirements.txt

weights/yolov5s.pt is a neural network used to implement the object detection step

try running: python yolov5_deepsort_db.py <path to frames of the video>

This will generate a .db file that contains all bbox and color information

try running: cureate_color_db.py

THis will convert the bbox and color information to a reader-friendly db with color labels
