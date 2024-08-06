# -*- coding: utf-8 -*-
"""
@author: Van Thiep <thiepne24u@gmail.com>
"""
"""Import necessary packages"""
import torch
import torchvision.transforms as transforms

classes_names = ['angry', 'disgusted', 'fearful', 'happy', 'neural', 'sad', 'surprised']

"""All task can tuning"""
DATASET = '/home/grasp/Downloads/archive2/data/'
TRAIN_DIR = f'{DATASET}train'
TEST_DIR = f'{DATASET}test'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKER = 4
IMAGE_SIZE = 380
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "Emotion.pt"

"""Transform for dataset"""
train_transform = transforms.Compose(
    [transforms.RandomCrop(IMAGE_SIZE),
     transforms.Resize(IMAGE_SIZE),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
)

test_transform = transforms.Compose(
    [transforms.Resize(IMAGE_SIZE),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
)
