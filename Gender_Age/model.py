# -*- coding: utf-8 -*-
"""
@author: Van Thiep <thiepne24u@gmail.com>
"""
import torch
import torch.nn as nn
import torchvision

class AgeModel(nn.Module):
    """Model recognition gender and age base on resnet50 model"""
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet50()

        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(in_features=1024,
                            out_features=512))

        # Branch of gender
        self.gender_ = nn.Sequential(nn.Dropout(0.2),
                                     nn.Linear(512, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 1))

        # Branch of age
        self.age_ = nn.Sequential(nn.Dropout(0.2),
                                  nn.Linear(512, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, 1))

    def forward(self, x):
        x = self.model(x)

        # Split 2 branch
        age = self.age_(x)
        gender = self.gender_(x)

        return gender, age
