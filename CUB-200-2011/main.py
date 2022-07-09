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

# ========================================
#               全局参数设定
# ========================================

temp = 20  # 知识蒸馏
alpha = 0.7  # 知识蒸馏
epochs = 1
lr = 0.005
weight_decay = 0.00005
momentum = 0.9
batch_size = 1
IMAGE_SIZE = 448
RE_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

data_path = 'data/CUB_200_2011'  # 设置数据路径
pretrained_model_path = 'data/resnet50-19c8e357.pth'  # 设置预训练模型路径
teacher_model = 'data/cub_epoch144.pth'  # 设置知识蒸馏模型路径
model_save_path = 'data/result/epoch%d_trainAcc%.3f_testAcc%.3f_resNet50.pth'  # 设置模型保存路径
train_loss_path = 'data/result/train_loss'  # 设置训练损失保存路径
test_loss_path = 'data/result/test_loss'  # 设置测试损失保存路径
train_acc_path = 'data/result/train_acc'  # 设置训练准确率保存路径
test_acc_path = 'data/result/test_acc'  # 设置测试准确率保存路径
lr_path = 'data/result/lr'  # 设置学习率保存路径

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

# ========================================
#               数据加载
# ========================================

train_dataset = CUB(
    data_path,
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
    data_path,
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

# ========================================
#               开始训练
# ========================================

print("using {} images for training, {} images fot testidation.".format(len(train_dataset), len(test_dataset)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
print('===================================================================================')

# -------------------------知识蒸馏--------------------------
teacher_net = MainNet(proposalN=proposalN, num_classes=200, channels=2048)
epoch = auto_load_resume(teacher_net, teacher_model, status='test')
teacher_net.to(device)
criterion = nn.KLDivLoss(reduction="batchmean")
# ----------------------------------------------------------

net = models.resnet50(pretrained=False)
# ----------------迁移学习部分--------------------------------
net.load_state_dict(torch.load(pretrained_model_path))
# ----------------------------------------------------------
# 这里需要先获取到预训练模型中的最后一个输出的全连接层然后再构建一个全连接层
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 200)  # 因为最后只需要划分出200个类别,所以要把原来的全连接层输出改为200
net.to(device)

# ------------------------标签平滑-----------------------------
loss_function = LabelSmoothingLoss(classes=200, smoothing=0.1)
# -------------------------------------------------------------

# loss_function = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.002, weight_decay=0.0001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)

best_acc = 0.0
train_num = len(train_dataset)
test_num = len(test_dataset)
test_acc_list = []
train_acc_list = []
test_loss_list = []
train_loss_list = []
learning_rate_list = []

for epoch in range(epochs):
    # train
    net.train()
    train_bar = tqdm(train_dataloader)
    for step, data in enumerate(train_bar):
        images, labels = data

        # -------------------------知识蒸馏---------------------------
        logits_t, _ = teacher_net(images.to(device), epoch, 0, 'test', device)[-2:]  # 老师模型的预测值
        logits = net(images.to(device))  # 学生模型的预测值
        loss_1 = loss_function(logits, labels.to(device))
        loss_2 = criterion(
            F.log_softmax(logits / temp, dim=1),
            F.softmax(logits_t / temp, dim=1)
        )
        a = F.softmax(logits_t / temp, dim=1)
        b = F.log_softmax(logits / temp, dim=1)
        loss = loss_1 * (1 - alpha) + loss_2 * alpha
        # ----------------------------------------------------------

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

    # ----------------------------学习率衰减--------------------------------
    learning_rate_list.append(optimizer.param_groups[0]['lr'])
    scheduler.step()
    # ---------------------------------------------------------------------

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
            tmp_train_loss = loss_function(train_outputs, train_labels.to(device))
            train_predict = torch.max(train_outputs, dim=1)[1]
            train_acc += torch.eq(train_predict, train_labels.to(device)).sum().item()
            train_loss += tmp_train_loss.item()
            train_bar.desc = "valid in train_dataset epoch[{}/{}]".format(epoch + 1, epochs)

        test_bar = tqdm(test_dataloader)
        for test_data in test_bar:
            test_images, test_labels = test_data
            test_outputs = net(test_images.to(device))
            tmp_test_loss = loss_function(test_outputs, test_labels.to(device))
            test_predict = torch.max(test_outputs, dim=1)[1]
            test_acc += torch.eq(test_predict, test_labels.to(device)).sum().item()
            test_loss += tmp_test_loss.item()
            test_bar.desc = "valid in test_dataset epoch[{}/{}]".format(epoch + 1, epochs)

    train_accurate = train_acc / train_num
    test_accurate = test_acc / test_num

    # 保存数据
    train_acc_list.append(train_accurate)
    test_acc_list.append(test_accurate)
    test_loss_list.append(test_loss / test_num)
    train_loss_list.append(train_loss / train_num)
    np.save(train_acc_path, train_acc_list)
    np.save(test_acc_path, test_acc_list)
    np.save(train_loss_path, train_loss_list)
    np.save(test_loss_path, test_loss_list)
    np.save(lr_path, learning_rate_list)

    if (test_accurate > best_acc):
        best_acc = test_accurate
        torch.save(net.state_dict(), model_save_path % (epoch + 1, train_accurate, test_accurate))

    tqdm.write('[epoch %d] train_loss: %.3f train_acc: %.3f test_loss:%.3f test_acc: %.3f'
               % (epoch + 1, train_loss / train_num, train_accurate, test_loss / test_num, test_accurate))
