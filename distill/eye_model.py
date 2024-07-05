import torch
import torch.nn as nn
from torchvision import models

class EyeModel(nn.Module):
    def __init__(self):
        super(EyeModel, self).__init__()
        
        # Load VGG-16 model and only keep the feature extraction part
        vgg16 = models.vgg16(pretrained=True)
        self.vgg16_features = nn.Sequential(*(list(vgg16.features.children()) + [nn.Flatten(), nn.Linear(512, 256)]))

        # Define the keypoints feature extraction part
        self.keypoint_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 68, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Define the gaze prediction head
        self.gaze_head = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output yaw and pitch angles
        )
        
        # Define the blink prediction head
        self.blink_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output blink probability
        )

    def forward(self, left_eye, right_eye, keypoints):
        # Extract features using VGG-16
        left_eye_vector = self.vgg16_features(left_eye) # shape: (n, 256)
        right_eye_vector = self.vgg16_features(right_eye) # shape: (n, 256)

        # Sum the eye vectors
        eye_vector_sum = left_eye_vector + right_eye_vector # shape: (n, 256)
        
        # Extract keypoint features
        keypoint_vector = self.keypoint_features(keypoints) # shape: (n, 64)

        # Concatenate eye vectors with keypoint vector for gaze prediction
        gaze_input = torch.cat((eye_vector_sum, keypoint_vector), dim=1) # shape: (n, 256 + 64)
        gaze_prediction = self.gaze_head(gaze_input)
        
        # Use eye vector sum for blink prediction
        blink_prediction = self.blink_head(eye_vector_sum)
        
        return gaze_prediction, blink_prediction