import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from PIL import Image
import matplotlib.pyplot as plt

# 定义超参数
num_epochs = 20
batch_size = 256
learning_rate = 0.1

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载 VOC2007 数据集
trainset = torchvision.datasets.VOCDetection(root='./data/VOCdevkit/VOC2007',
                                              image_set='trainval',
                                              download=True,
                                              transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# 加载 CIFAR100 数据集
dataset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# 划分数据集
pretrain_size = int(0.5 * len(dataset))
train_size = len(dataset) - pretrain_size
pretrainset, trainset = random_split(dataset, [pretrain_size, train_size])

# 创建数据加载器
pretrainloader = torch.utils.data.DataLoader(pretrainset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# 定义 ResNet-18 网络模型
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 预训练模型
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(pretrainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 计算模型在测试集上的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Epoch: {epoch + 1}/{num_epochs}, Pretrain Loss: {running_loss / len(pretrainloader):.4f}, Test Accuracy: {correct / total:.4f}')

# 保存预训练好的模型
torch.save(model.state_dict(), 'pretrained_model.pth')

# 加载预训练好的模型
model.load_state_dict(torch.load('pretrained_model.pth'))

model = torchvision.models.resnet18(weights=None)
model.fc = nn.Identity()
model = model.to(device)

# 定义投影头
projector = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 128)
).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(model.parameters()) + list(projector.parameters()), lr=0.1)
temperature = 0.5

# 训练模型
Train_Loss = []
Test_Accuracy = []

for epoch in range(num_epochs):
    for i, data in enumerate(trainloader):
        inputs, _ = data
        inputs = inputs.to(device)
        batch_size = inputs.size(0)

        # 数据增强
        inputs1 = transform(inputs)
        inputs2 = transform(inputs)

        # 前向传播
        features1 = model(inputs1)
        features2 = model(inputs2)
        projections1 = projector(features1)
        projections2 = projector(features2)

        # 计算损失
        logits11 = torch.mm(projections1, projections1.t()) / temperature
        logits22 = torch.mm(projections2, projections2.t()) / temperature
        logits12 = torch.mm(projections1, projections2.t()) / temperature
        logits21 = torch.mm(projections2, projections1.t()) / temperature

        mask = torch.eye(batch_size).to(device)
        logits11 = logits11.masked_fill(mask == 1, float('-inf'))
        logits22 = logits22.masked_fill(mask == 1, float('-inf'))

        labels1 = torch.arange(batch_size).to(device)
        labels2 = torch.arange(batch_size).to(device)

        loss1 = criterion(logits12, labels1) + criterion(logits21, labels2)
        loss2 = criterion(logits11, labels1) + criterion(logits22, labels2)
        loss = loss1 + loss2

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算模型在测试集上的准确率
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    Train_Loss.append(running_loss / len(trainloader))
    Test_Accuracy.append(correct / total)

    print(f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(trainloader):.4f}, Test Accuracy: {correct / total:.4f}')


# 绘制训练损失曲线
plt.figure()
plt.plot(range(len(Train_Loss)), Train_Loss, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# 绘制测试准确率曲线
plt.figure()
plt.plot(range(len(Test_Accuracy)), Test_Accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()
plt.show()


# 图像分类示例
def classify(index):
    # 加载训练好的模型
    model.load_state_dict(torch.load('trained_model.pth'))

    # 设置模型为评估模式
    model.eval()

    # 随机选择一张图片
    image, label = testset[index]
    image = image.unsqueeze(0).to(device)

    # 使用模型对图片进行分类
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)

    # 显示图片和分类结果
    # print(f'Predicted: {testset.classes[predicted[0]]}')
    original_label = testset.classes[label]
    print(f'Original label: {original_label}', ',\t', f'Predicted: {testset.classes[predicted[0]]}')
    img = Image.fromarray(testset.data[i])
    new_size = (img.width * 2, img.height * 2)
    img = img.resize(new_size, Image.ANTIALIAS)
    img.show()


indexs = [4 * i for i in range(1, 5)]

for i in indexs:
    classify(i)
