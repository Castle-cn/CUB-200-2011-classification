import numpy as np
from tqdm import tqdm
from dataset import CUB
import torch
from teacher_model import MainNet, auto_load_resume, proposalN
import torch.nn.functional as F
from torchvision import models, transforms
from labelsmothing import LabelSmoothingLoss
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

batch_size = 16
IMAGE_SIZE = 448
RE_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

path = 'data/CUB_200_2011'
trained_model_path = 'data/epoch24_trainAcc1.000_testAcc0.866_resNet50.pth'

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RE_SIZE, RE_SIZE)),
    transforms.RandomCrop(IMAGE_SIZE, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RE_SIZE, RE_SIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

train_dataset = CUB(
    path,
    train=True,
    transform=train_transforms,
    target_transform=None
)
# print(len(train_dataset))
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
)

test_dataset = CUB(
    path,
    train=False,
    transform=test_transforms,
    target_transform=None
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = models.resnet50(pretrained=False)
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 200)
net.load_state_dict(torch.load(trained_model_path))
net.to(device)

train_num = len(train_dataset)
test_num = len(test_dataset)

# test
net.eval()
test_acc = 0.0  # 累计测试集中的所有正确答对的个数
train_acc = 0.0  # 累计训练集中所有正确答对的个数
test_loss = 0.0  # 累计测试集中所有误差
train_loss = 0.0  # 累积训练集中所有误差
with torch.no_grad():
    train_bar = tqdm(train_dataloader)
    for train_data in train_bar:
        train_images, train_labels = train_data
        train_outputs = net(train_images.to(device))
        train_predict = torch.max(train_outputs, dim=1)[1]
        train_acc += torch.eq(train_predict, train_labels.to(device)).sum().item()

    test_bar = tqdm(test_dataloader)
    for test_data in test_bar:
        test_images, test_labels = test_data
        test_outputs = net(test_images.to(device))
        test_predict = torch.max(test_outputs, dim=1)[1]
        test_acc += torch.eq(test_predict, test_labels.to(device)).sum().item()

train_accurate = train_acc / train_num
test_accurate = test_acc / test_num

tqdm.write('train_acc: %.3f  test_acc: %.3f' % (train_accurate, test_accurate))
