import cv2
import json
import numpy as np
from rt_gene import gaze_tools as gaze_tools
from rt_gene.tracker_generic import TrackedSubject

with open("./gaze_result.txt") as f:
    lines = f.readlines()

for line in lines:
    elems = line.split(" # ")
    image = cv2.imread(f"/home/jiangmuye/dataset/12/{elems[0]}")
    box = json.loads(elems[1])
    landmarks = json.loads(elems[2])
    gaze_est = json.loads(elems[3])
    subject = TrackedSubject(np.array(box), gaze_tools.crop_face_from_image(image, box), np.array(landmarks))
    le_c, re_c, le_bb, re_bb = TrackedSubject.get_eye_image_from_landmarks(subject, (60, 36))
    print(f"Processing: {elems[0]}")
    cv2.imwrite(f"/home/jiangmuye/dataset/12/left/{elems[0]}", le_c)
    cv2.imwrite(f"/home/jiangmuye/dataset/12/right/{elems[0]}", re_c)