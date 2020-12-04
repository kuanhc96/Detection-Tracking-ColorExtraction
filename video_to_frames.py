# USAGE
# python read_frames_slow.py --video videos/jurassic_park_intro.mp4

# import the necessary packages

import numpy as np
import argparse
#import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to input video file")
ap.add_argument("-o", "--output_path", required=True,
        help="path to output directory")
args = vars(ap.parse_args())

# open a pointer to the video stream and start the FPS timer
stream = cv2.VideoCapture(args["video"])
output = args["output_path"]
frame_num = 0
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video file stream
    (grabbed, frame) = stream.read()
    print("frame", frame_num)
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    
    if frame_num < 10:
        prefix = "0000000"
    elif frame_num < 100:
        prefix = "000000"
    elif frame_num < 1000:
        prefix = "00000"
    elif frame_num < 10000:
        prefix = "0000"
    elif frame_num < 100000:
        prefix = "000"
    elif frame_num < 1000000:
        prefix = "00"
    elif frame_num < 10000000:
        prefix = "0"
    else:
        prefix = ""
    cv2.imwrite(output + prefix + str(frame_num) + ".jpg", frame)
    frame_num += 1



# do a bit of cleanup
stream.release()

