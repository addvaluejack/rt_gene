# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

import os

import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

from rt_gene.estimate_gaze_base import GazeEstimatorBase
from eye_model import EyeModel

class DistilledGazeEstimator(GazeEstimatorBase):
    def __init__(self, device_id_gaze, model_files):
        super(DistilledGazeEstimator, self).__init__(device_id_gaze, model_files)

        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "8"
        tqdm.write("PyTorch using {} threads.".format(os.environ["OMP_NUM_THREADS"]))

        self._transform = transforms.Compose([lambda x: cv2.resize(x, dsize=(60, 36), interpolation=cv2.INTER_CUBIC),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        self._models = []
        for ckpt in self.model_files:
            try:
                _model = EyeModel()
                _model.load_state_dict(torch.load(ckpt)['model_state_dict'])
                _model.to(self.device_id_gazeestimation)
                _model.eval()
                self._models.append(_model)
            except Exception as e:
                print("Error loading checkpoint", ckpt)
                raise e

        tqdm.write('Loaded ' + str(len(self._models)) + ' model(s)')

    def estimate_gaze_twoeyes(self, inference_input_left_list, inference_input_right_list, landmarks_list):
        transformed_left = torch.stack(inference_input_left_list).to(self.device_id_gazeestimation)
        transformed_right = torch.stack(inference_input_right_list).to(self.device_id_gazeestimation)
        landmarks = torch.as_tensor(landmarks_list).to(self.device_id_gazeestimation, dtype=torch.float32)

        result = [model(transformed_left, transformed_right, landmarks)[0].detach().cpu() for model in self._models]
        result = torch.stack(result, dim=1)
        result = torch.mean(result, dim=1).numpy()
        result[:, 1] += self._gaze_offset
        return result

    def input_from_image(self, cv_image):
        return self._transform(cv_image)
