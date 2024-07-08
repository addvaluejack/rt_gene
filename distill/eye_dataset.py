import cv2
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms

eye_transform = transforms.Compose([lambda x: cv2.resize(x, dsize=(60, 36), interpolation=cv2.INTER_CUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

class EyeDataset(Dataset):
    def __init__(self):
        self.eye_result = []
        eye_result_dict = {}

        with open("./gaze_result.txt") as f:
            gaze_lines = f.readlines()
        for line in gaze_lines:
            elems = line.split(" # ")
            eye_result_dict[elems[0]] = {
                "filename": elems[0],
                "box": json.loads(elems[1]),
                "landmarks": json.loads(elems[2]),
                "gaze_est": json.loads(elems[3])[0] 
            }
        with open("./blink_result.txt") as f:
            blink_lines = f.readlines()
        for line in blink_lines:
            elems = line.split(" # ")
            eye_result_dict[elems[0]]["blink_prob"] = json.loads(elems[1])[0]
        
        self.eye_result = [value for _, value in eye_result_dict.items()]

    def __getitem__(self, index):
        leye_image = eye_transform(cv2.imread(f"/home/jiangmuye/dataset/12/left/{self.eye_result[index]['filename']}"))
        reye_image = eye_transform(cv2.imread(f"/home/jiangmuye/dataset/12/right/{self.eye_result[index]['filename']}"))
        landmarks = torch.tensor(self.eye_result[index]["landmarks"], dtype=torch.float32)
        gaze_est = torch.tensor(self.eye_result[index]["gaze_est"])
        blink_prob = torch.tensor(self.eye_result[index]["blink_prob"])
        return leye_image, reye_image, landmarks, gaze_est, blink_prob

    def __len__(self):
        return len(self.eye_result)