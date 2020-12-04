import sqlite3
from Bounding_Boxes import Bounding_Boxes
import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from ImageProcessing import SaltPepperNoise, findSignificantContour
from ColorLabeler import ColorLabeler
from Yolov5_Evaluator import Yolov5Evaluator
from utils.datasets import *
from utils.utils import *
from models.experimental import *
from detector import build_detector
from deep_sort import build_tracker
from tracking_utils.draw import draw_boxes
from tracking_utils.parser import get_config
from tracking_utils.log import get_logger
from tracking_utils.io import write_results


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        
        # create database -- will throw error if database already exists
        self.conn = sqlite3.connect('test4.db')
        self.c = self.conn.cursor()

        # datavase schema
        self.c.execute("""CREATE TABLE bounding_boxes (
                    id TEXT,
                    frame_id TEXT,
                    xl TEXT,
                    yl TEXT,
                    xr TEXT,
                    yr TEXT,
                    red INTEGER,
                    green INTEGER,
                    blue INTEGER
                )""")

        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")
        self.cl = ColorLabeler()

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        # construct object detector
        yolov5 = Yolov5Evaluator("weights/yolov5s.pt") 
        yolov5.prepare_detector()
        self.detector = yolov5
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
    
    def __del__(self):
        self.conn.close()

    # insert bbox and color information
    def insert_bbox(self, identity, frame_num, bbox, rgb):
        # create ascii-sortable frame IDs
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
        with self.conn:
            for i in range(len(identity)):
                self.c.execute("INSERT INTO bounding_boxes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (str(identity[i]), prefix+ str(frame_num) +".jpg", str(bbox[i][0]), str(bbox[i][1]), str(bbox[i][2]), str(bbox[i][3]), 
                            rgb[i][0], rgb[i][1], rgb[i][2]))

                
    def __enter__(self):
        frame0 = cv2.imread(os.path.join(self.video_path , str("00000000.jpg")))
        im_width = frame0.shape[1]
        im_height = frame0.shape[0]
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 60, (im_width, im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        
        # load frames from video
        dataset = LoadImages(self.video_path, img_size=1920)
        
        for idx_frame, (path, ori_im, im0s, vid_cap) in enumerate(dataset):

            start = time.time()

            # object detection detection -- load_detections_and_confidence only returns detections of the "person" class
            # bbox format: x1, y1, x2, y2
            bbox, cls_conf = self.detector.load_detections_and_confidence([ori_im, im0s])
            
            # skip analysis if no detections found
            if len(bbox) == 0:
                continue

            # convert x1, y1, x2, y2 to xc, yc, w, h, which is the format accepted by the tracking algorithm
            bbox_xywh = np.zeros((len(bbox), 4))
            for i in range(len(bbox)):
                width = bbox[i][1][0] - bbox[i][0][0]
                height = bbox[i][1][1] - bbox[i][0][1]
                bbox_xywh[i][0] = bbox[i][1][0] - width / 2
                bbox_xywh[i][1] = bbox[i][1][1] - height / 2
                bbox_xywh[i][2] = width
                bbox_xywh[i][3] = height

            bbox_xywh = np.array(bbox_xywh)

            # do tracking
            ori_im = ori_im[:,:,::-1].transpose(1, 2, 0)
            # output format: x1, y1, x2, y2, ID
            outputs = self.deepsort.update(bbox_xywh, cls_conf, np.array(ori_im)) 

            # process bbox information
            if len(outputs) > 0:
                # get colors
                colors = list()
                # instantiate edge detector
                edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("edge_model.yml")
                for output in outputs:
                    x1 = output[0]
                    y1 = output[1]
                    x2 = output[2]
                    y2 = output[3]
                    # zoom into bbox for processing
                    subframe = ori_im[y1: y2, x1: x2]
                    # apply Gaussian blur -- convolution filter size cannot be dynamically computed
                    bbox_blurred = cv2.GaussianBlur(subframe, (11,11), 0)
                    blurred_float = bbox_blurred.astype(np.float32) / 255.0
                    
                    try:
                        # detect and highlight significant edges in bbox
                        edges = edgeDetector.detectEdges(blurred_float) * 255.0
                        edges_ = np.asarray(edges, np.uint8)
                        SaltPepperNoise(edges_)

                        # contour generation
                        contour = findSignificantContour(edges_)
                        if contour is None:
                            # contour generation failed
                            colors.append([None, None, None])
                        else:
                            # get mean RGB in the contour
                            lab = cv2.cvtColor(bbox_blurred, cv2.COLOR_BGR2LAB)
                            rgb_mean = self.cl.get_rgb(lab, contour)
                            # record color
                            colors.append(rgb_mean)
                    except:
                        colors.append([None, None, None])
                colors = np.array(colors)

                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                self.insert_bbox(identities, idx_frame, bbox_xyxy, colors)

                # draw boxes onto the frame
                ori_im = draw_boxes(im0s, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)
            
            idx_frame += 1
            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_deepsort", type=str, default="./tracking_configs/deep_sort.yaml")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./tracking_output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
