import sqlite3
import argparse
import cv2
import random
from ColorLabeler import ColorLabeler as CL

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--id", required=True, type=str,
	help="person id to be accessed")

args = vars(ap.parse_args())

conn = sqlite3.connect('bounding_boxes.db')

c = conn.cursor()

def get_bbox(id):
    c.execute("SELECT * "
              "FROM bounding_boxes "
              "WHERE id=:ID", {'ID': id})
    return c.fetchall()



def get_color_prediction(id):
    c.execute("SELECT AVG(red) as R, AVG(green) as G, AVG(blue) as B "
              "FROM bounding_boxes "
              "WHERE id=:ID "
              "GROUP BY id ", {'ID': id})
    return c.fetchall()


def get_color_prediction_and_std():
    c.execute("SELECT id, AVG(red) as R, AVG(green) as G, AVG(blue) as B, STDDEV(red) as std_R, STDDEV(green) as std_green, STDDEV(blue) as std_blue "
              "from BBOX_WITH_COLORS "
              "WHERE red IS NOT NULL AND green IS NOT NULL AND blue IS NOT NULL "
              "GROUP BY id ")
    return c.fetchall()


bbox_tuple = get_bbox(args["id"])
print(bbox_tuple)
sample = random.choices(bbox_tuple, k=10)

for i, bbox in enumerate(sample):
    image = cv2.imread("frames/" + bbox[1])
    x1 = int(bbox[2])
    y1 = int(bbox[3])
    x2 = int(bbox[4])
    y2 = int(bbox[5])
    cropped = image[y1:y2, x1:x2]
    cv2.imshow(bbox[0] + " " + str(i), cropped)

color_prediction = get_color_prediction(args["id"])

color_labeler = CL()

color_name = color_labeler.match_color(color_prediction)

print("prediction:", color_name)

while True:
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()