from Base_Evaluator import BaseEvaluator
import argparse
import json
import torch.backends.cudnn as cudnn
import threading
from models.experimental import *
from utils.datasets import *
from utils.utils import *
import numpy as np
import cv2
import pandas as pd
import time

class Yolov5Evaluator(BaseEvaluator):
    def __init__(self, weights, iou_thresh=0.1, detector_name="Yolov5", config=None):
        # Initialize
        self.device = torch_utils.select_device('')
        self.input_size = 640
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        super(Yolov5Evaluator, self).__init__(iou_thresh, detector_name, weights, config=config)

    def read_class_names(self, class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    def read_json(self, filename):
        """
        Decodes a JSON file & returns its content.
        Raises:
            FileNotFoundError: file not found
            ValueError: failed to decode the JSON file
            TypeError: the type of decoded content differs from the expected (list of dictionaries)
        :param filename: [str] name of the JSON file
        :return: [list] list of the annotations
        """
        if not os.path.exists(filename):
            raise FileNotFoundError("File %s not found." % filename)
        try:
            with open(filename, 'r') as _f:
                _data = json.load(_f)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode {filename}.")
        if not isinstance(_data, list):
            raise TypeError(f"Decoded content is {type(_data)}. Expected list.")
        if len(_data) > 0 and not isinstance(_data[0], dict):
            raise TypeError(f"Decoded content is {type(_data[0])}. Expected dict.")
        return _data

    # get the ground-truth bbox coordinates from the json annotation files
    def load_ground_truth_coordinates(self, filename, view=None): # filename format is 0000XXXX.png, XXXX is a timestamp
        if view == None:
            raise TypeError("view cannot be None")
        annotations = self.read_json(filename)
        coordinates = list()
        for annotation in annotations:
            bbox = annotation['views'][view-1] # 0 - 6, representing C1 - C 7
            xmin = bbox['xmin']
            ymin = bbox['ymin']
            xmax = bbox['xmax']
            ymax = bbox['ymax']
            if (xmin, ymin, xmax, ymax) == (-1, -1, -1, -1): # person not present
                continue
            coordinate = list()
            coordinate.append((xmin, ymin))
            coordinate.append((xmax, ymax))
            coordinates.append(coordinate)

        return coordinates

    def prepare_detector(self):
        # Load model
        self.detector = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.input_size = check_img_size(self.input_size, s=self.detector.stride.max())  # check img_size
        if self.half:
            self.detector.half()  # to FP16
        # Get names and colors
        names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, self.input_size, self.input_size), device=self.device)  # init img
        _ = self.detector(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once

    def load_detections_and_confidence(self, args):
        original_image = args[0]
        im0 = args[1]
        original_image = torch.from_numpy(original_image.copy()).to(self.device)
        original_image = original_image.half() if self.half else original_image.float()  # uint8 to fp16/32
        original_image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if original_image.ndimension() == 3:
            original_image = original_image.unsqueeze(0)
        
        # Inference
        start = time.time()
        t1 = torch_utils.time_synchronized()
        pred = self.detector(original_image, augment=False)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, 0.3, 0.5, classes=None, agnostic=False)[0]
        if pred is None:
            return [], []
        t2 = torch_utils.time_synchronized()
        end = time.time()
        image_time_elapsed = end-start
        self.times.append(image_time_elapsed)
        
        pred[:, :4] = scale_coords(original_image.shape[2:], pred[:, :4], im0.shape).round()
        
        detections = list()
        confidence = list()
        
        # reformat bboxes
        for det in pred:
            det = det.detach().numpy()
            if int(det[5]) == 0:
                x1 = det[0]
                y1 = det[1]
                x2 = det[2]
                y2 = det[3]
                coordinate = list()
                coordinate.append((x1, y1))
                coordinate.append((x2, y2))
                detections.append(coordinate)
                confidence.append(det[4])
        
        return detections, confidence
    
    def evaluate_one_set(self, view):
            time_stamp = 0
            dataset = LoadImages('../Wildtrack_dataset/Image_subsets/C' + str(view), img_size=self.input_size)
            self.prepare_detector()
            
            for path, img, im0s, vid_cap in dataset:
                if time_stamp > 1995:
                    break
                detections, confidence = self.load_detections_and_confidence([img, im0s])
                print(detections)
                self.num_detections.append(len(detections))
                if int(time_stamp / 10) == 0: # single digit 000X
                    stamp = "000" + str(time_stamp)
                elif int(time_stamp / 100) == 0: # two digits 00XX
                    stamp = "00" + str(time_stamp)
                elif int(time_stamp / 1000) == 0: # three digits 0XXX
                    stamp = "0" + str(time_stamp)
                else: # XXXX
                    stamp = str(time_stamp)
                print("processing C"+str(view)+"0000"+stamp)
                self.imageIDs.append("C"+str(view)+"0000"+stamp+".png")
                ground_truth_file = "../Wildtrack_dataset/annotations_positions/0000"+stamp+".json"
                ground_truth_coordinates = self.load_ground_truth_coordinates(ground_truth_file, view=view)
                self.classify_detections(detections, confidence, ground_truth_coordinates, "C"+str(view)+"0000"+stamp+".png")
                time_stamp += 5

    def evaluate(self):
        #x = threading.Thread(target=thread_function, args=(1,))
        #threads = list()
        for view in range(1, 8): # examine 1 video at a time
            #t = threading.Thread(target=self.evaluate_one_set, args=(view,))
            #t.start()
            #threads.append(t)
            self.evaluate_one_set(view)
            '''
            time_stamp = 0
            dataset = LoadImages('../Wildtrack_dataset/Image_subsets/C' + str(view), img_size=self.input_size)
            self.prepare_detector()
            for path, img, im0s, vid_cap in dataset:
                if time_stamp > 1995:
                    break
                detections, confidence = self.load_detections_and_confidence([img, im0s])
                if int(time_stamp / 10) == 0: # single digit 000X
                    stamp = "000" + str(time_stamp)
                elif int(time_stamp / 100) == 0: # two digits 00XX
                    stamp = "00" + str(time_stamp)
                elif int(time_stamp / 1000) == 0: # three digits 0XXX
                    stamp = "0" + str(time_stamp)
                else: # XXXX
                    stamp = str(time_stamp)
                print("processing C"+str(view)+"0000"+stamp)
                ground_truth_file = "../Wildtrack_dataset/annotations_positions/0000"+stamp+".json"
                ground_truth_coordinates = self.load_ground_truth_coordinates(ground_truth_file, view=view)
                self.classify_detections(detections, confidence, ground_truth_coordinates, "C"+str(view)+"0000"+stamp+".png")
                time_stamp += 5
            '''

        #for thread in threads:
            #thread.join()
        
        self.timer_dataframe["ImageID"] = self.imageIDs
        self.timer_dataframe["num_detections"] = self.num_detections
        self.timer_dataframe["time"] = self.times
        plt.plot(self.num_detections, self.times, 'ro')
        plt.title("num_detections vs time")
        plt.xlabel("number of detections")
        plt.ylabel("time elapsed")
        plt.savefig("times plot.png", format='png')
        self.timer_dataframe.to_csv("times.csv",index=False)

        self.calculate_final_stats()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run evaluation with iou threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.1)
    args = parser.parse_args()
    evaluator = Yolov5Evaluator("weights/yolov5s.pt", iou_thresh=args.iou_thresh, detector_name="Yolov5")
    evaluator.evaluate()

