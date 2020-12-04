import pandas as pd
import numpy as np
import cv2

class BaseEvaluator(object):
    def __init__(self, iou_thresh, detector_name, weights, config=None):
        self.detector = None
        self.detector_name = detector_name
        self.weights = weights
        self.config = config
        self.confidence_thresh = 0.3
        # the next 4 attributes (columns) to be added to the overall dataframe
        self.acc_TP = list()
        self.acc_FP = list()
        self.precision = list()
        self.recall = list()
        self.current_acc_TP = 0 # accumulated number of TP seen up to current row
        self.current_acc_FP = 0 # accumulated number of FP seen up to current row

        # image being examined, coordinate of a prediction (detection), confidnce score of a prediction, TP = 1 if prediction is tru positive, 0 otherwise, FP = 1 if prediction is false positive, 0 otherwise
        self.dataframe = pd.DataFrame(columns=("ImageID", "coordinate", "confidence", "TP", "FP"))
        self.timer_dataframe = pd.DataFrame(columns=("ImageID", "num_detections", "time"))
        self.imageIDs = list()
        self.num_detections = list()
        self.times = list()
        # total number of ground truth bboxes = TP + FN (a ground-truth bbox is either detected or not detected), used in calculating recall
        self.total_gt_boxes = 0
        self.total_image_precision = 0
        self.total_recall = 0
        self.total_F1_score = 0
        self.total_images = 0
        self.iou_thresh = iou_thresh 
        

    def load_ground_truth_coordinates(self, filename, view=None):
        raise NotImplementedError

    def load_detections_and_confidence(self, image):
        raise NotImplementedError

    def prepare_detector(self):
        raise NotImplementedError
    
    def evaluate(self):
        raise NotImplementedError

    def classify_detections(self, detections, confidence, ground_truth_coordinates, image_name):
        num_ground_truth = len(ground_truth_coordinates)
        self.total_gt_boxes += num_ground_truth
        # create dataframe for each image, storing information regarding each prediction it contains 
        #Later appended to the "overall" dataframe
        image_dataframe = pd.DataFrame(columns=("ImageID", "coordinate", "confidence", "TP", "FP"))
        
        image_dataframe["ImageID"] = [image_name] * len(detections)
        image_dataframe["coordinate"] = detections
        image_dataframe["confidence"] = confidence
        image_dataframe["TP"] = [0] * len(detections)
        image_dataframe["FP"] = [0] * len(detections)
        self.get_stats(ground_truth_coordinates, image_dataframe)
        precision, recall, F1 = self.get_image_precision_recall_F1(image_dataframe, num_ground_truth)

        self.total_image_precision += precision
        self.total_recall += recall
        self.total_F1_score += F1
        self.total_images += 1
        self.dataframe = pd.concat([self.dataframe, image_dataframe], ignore_index=True)
        #print(image_dataframe)


    # calculate the overlapping area between prediction bbox and ground truth bbox
    def calc_overlap(self, gt_coordinate, p_coordinate):
        gt_lower = gt_coordinate[0]
        gt_upper = gt_coordinate[1]
        p_lower = p_coordinate[0]
        p_upper = p_coordinate[1]

        gt_area = (gt_upper[0] - gt_lower[0]) * (gt_upper[1] - gt_lower[1])
        p_area = (p_upper[0] - p_lower[0]) * (p_upper[1] - p_lower[1])
        '''
        def area(a, b):  # returns None if rectangles don't intersect
            dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
            dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
            if (dx>=0) and (dy>=0):
                return dx*dy
        '''

        dx = min(gt_upper[0], p_upper[0]) - max(gt_lower[0], p_lower[0])
        dy = min(gt_upper[1], p_upper[1]) - max(gt_lower[1], p_lower[1])
        if (dx >=0) and (dy>=0):
            return dx*dy, dx*dy / (gt_area + p_area - dx*dy)
        else:
            return 0, 0

    # checks to see if a detection is TP or FP, and fills the TP and FP field in dataframe
    def get_stats(self, ground_truth, dataframe):

        # if a prediction is "seen" that means it is already matched to a ground-truth bbox and should not be reassigned to another ground-truth bbox
        # since, of course, one prediction is meant to only predict one groud-truth
        seen_predictions = list()

        # find matches for the ground-truth bboxes. A matching prediction bbox is one that is closest to it
        # if a ground truth bbox cannot find a match, that indicates an FN prediction occurred
        for gt_coordinate in ground_truth:
            # these values should change at the end of the ensuing for loop if a match is found
            max_area, max_area_percent = 0, 0
            max_area_prediction = None
            max_index = - 1

            for row in dataframe.itertuples(): # find the prediction corresponding to current ground-truth bbox
                # each "coordinate" represents a unique prediction (detection) that the algorithm made
                p_coordinate = row.coordinate
                index = row.Index # store the position of this prediction
                if p_coordinate not in seen_predictions: # the prediction is not matched with a ground-truth bbox

                    area, area_percent = self.calc_overlap(gt_coordinate, p_coordinate) # check overlap between the prediction and ground-truth bbox
                    if area > max_area: # the ground-truth and prediction pair with the greatest overlap (closest) compared to all other predictions is said to be a match
                        max_area = area
                        max_area_percent = area_percent
                        max_area_prediction = p_coordinate
                        max_index = index # get the index of the matching prediction
            #print("GT:", gt_coordinate, "P", max_area_prediction, "percent", max_area_percent)
            if max_area_prediction != None: # this means that the prediction was matched with a ground-truth bbox
                seen_predictions.append(max_area_prediction) # the prediction is now "seen" and should not be used again to make another match
                if max_area_percent >= self.iou_thresh: # this means that the prediction is valid
                    dataframe.iloc[max_index, 3] += 1 # 3rd entry is TP
                else:
                    dataframe.iloc[max_index, 4] += 1 # 4th entry is

    def get_image_precision_recall_F1(self, image_dataframe, num_ground_truth):
        total_TP = np.sum(image_dataframe['TP'])
        total_FP = np.sum(image_dataframe['FP'])
        if total_TP == 0 and total_FP == 0:
            return 1, 0, 0
        elif total_TP == 0:
            return 1, 0, 0
        image_precision = total_TP / (total_FP + total_TP)
        image_recall = total_TP / num_ground_truth
        return image_precision, image_recall, 2 * image_precision * image_recall / (image_precision + image_recall)

    def calculate_final_stats(self):
        ### Collecting data ###
        print(self.dataframe)
        self.dataframe.sort_values(by=['confidence'], inplace=True,ascending=False, ignore_index=True) # sort by confidence

        for row in self.dataframe.itertuples(): # iterate over the rows to fill acc_TP, acc_FP, precision, and recall values
            self.current_acc_TP += self.dataframe.iloc[row.Index, 3] # 3rd column is TP
            self.current_acc_FP += self.dataframe.iloc[row.Index, 4] # 4th column is FP
            if self.current_acc_TP == 0 and self.current_acc_FP == 0: # edge case, when TP and FP are both 0, but FN is not, then let precision be 1
                current_precision = 1
            else:
                current_precision = self.current_acc_TP * 1.0 / (self.current_acc_TP + self.current_acc_FP) # precision up to current row
            self.precision.append(current_precision) # store current precision
            current_recall = self.current_acc_TP * 1.0 / self.total_gt_boxes # recall up to current row
            self.recall.append(current_recall) # store current recall
            self.acc_TP.append(self.current_acc_TP) # store current accumulated TP
            self.acc_FP.append(self.current_acc_FP) # store current accumulated FP

        # add columns into dataframe
        self.dataframe["acc_TP"] = self.acc_TP
        self.dataframe["acc_FP"] = self.acc_FP
        self.dataframe["precision"] = self.precision
        self.dataframe["recall"] = self.recall

        ##### Calculating stats ######

        # sort on recall values, increasing in value
        self.dataframe.sort_values(by=['recall'], inplace=True, ascending=True, ignore_index=True)
        print(self.dataframe)

        # max recall, beyond which precision will be 0
        max_recall = np.max(self.dataframe['recall'])

        # get only the precision and recall columns to calculate AP
        sub_dataframe = self.dataframe[['precision', 'recall']]
        for r in np.arange(0, 1.01, 0.1): # insert 11 points, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 into sub_dataframe
            sub_dataframe = sub_dataframe.append({'precision':-1, 'recall':r}, ignore_index=True)

        sub_dataframe.sort_values(by=['recall'], inplace=True, ascending=False, ignore_index = True) # sort 11 points into sub_dataframe, let the recall to be in DESCENDING order so that interpolation can work backwards

        total_precision = 0
        current_max_precision = 0
        for row in sub_dataframe.itertuples():
            record = False
            if sub_dataframe.iloc[row.Index, 0] == -1: # one of the inserted values to be recorded for 11 point average
                record = True
            if sub_dataframe.iloc[row.Index, 1] > max_recall: # values that exceed the available max recall value should have precision = 0
                sub_dataframe.iloc[row.Index, 0] = 0
            else:
                if sub_dataframe.iloc[row.Index, 0] > current_max_precision: # update current max precision. THis is the value that in-between values will be interpolated to
                    current_max_precision = sub_dataframe.iloc[row.Index, 0]
                else:
                    sub_dataframe.iloc[row.Index, 0] = current_max_precision # interpolate
            if record:
                total_precision += sub_dataframe.iloc[row.Index, 0] # add the precision of the inserted value to the total

        print(sub_dataframe)
        # average total stats
        AP = total_precision / 11
        precision = self.total_image_precision / self.total_images
        recall = self.total_recall / self.total_images
        F1 = self.total_F1_score / self.total_images
        print("AP =", AP)
        print("Avg Precision =", precision)
        print("Avg Recall =", recall)
        print("Avg F1 score =", F1)

        try:
            f = open("stats.csv", "x") # file does not exist, error if file exists
            f.write("model,iou,AP,precision,recall,F1\n")
        except FileExistsError:
            f = open("stats.csv", "a") # file already exists, append to it
        f.write(self.detector_name+","+str(self.iou_thresh)+","+str(AP)+","+str(precision)+","+str(recall)+","+str(F1)+"\n")



