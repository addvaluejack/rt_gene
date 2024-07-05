import cv2
import torch
from torchvision import transforms
from eye_model import EyeModel


eye_transform = transforms.Compose([lambda x: cv2.resize(x, dsize=(60, 36), interpolation=cv2.INTER_CUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

model = EyeModel()
left_eye = torch.stack([eye_transform(cv2.imread("/home/jiangmuye/dataset/12/left/1000.jpg")), eye_transform(cv2.imread("/home/jiangmuye/dataset/12/left/1001.jpg"))])
right_eye = torch.stack([eye_transform(cv2.imread("/home/jiangmuye/dataset/12/right/1000.jpg")), eye_transform(cv2.imread("/home/jiangmuye/dataset/12/right/1001.jpg"))])
keypoints = torch.stack([torch.randn(68, 2), torch.randn(68, 2)]) # Example keypoints (68 points with x, y coordinates)

gaze_prediction, blink_prediction = model(left_eye, right_eye, keypoints)
print("Gaze Prediction:", gaze_prediction.shape)
print("Blink Prediction:", blink_prediction.shape)
