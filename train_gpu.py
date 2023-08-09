import torch
import os
from torch.utils.data import DataLoader
from torch import nn, optim

from lenet import Lenet
from dataset import MyData

torch.manual_seed(1)
torch.cuda.manual_seed(1)

# 参数初始化
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
         torch.nn.init.kaiming_normal_(m.weight.data)
         torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
         torch.nn.init.kaiming_normal_(m.weight)
         torch.nn.init.constant_(m.bias, 0.0)


def main():
    batch_size = 32
    lr = 1e-2
    EPOCHS = 25
    #绝对路径设置为自己的文件位置，注意test中路径也要设置
    train_dir = 'D://Pycharm//DeepLearning//Gesture_Recognition//data//train//'
    test_dir = 'D://Pycharm//DeepLearning//Gesture_Recognition//data//test//'
    DEVICE = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    CAR_train = MyData(data_dir=train_dir)
    CAR_train = DataLoader(CAR_train, batch_size=batch_size, shuffle=True, num_workers=0)
    CAR_test = MyData(train=False, data_dir=test_dir)
    CAR_test = DataLoader(CAR_test, batch_size=1, shuffle=False, num_workers=0)

    model = Lenet().to(DEVICE)
    #model.apply(weight_init)
    f_loss = nn.CrossEntropyLoss().to(DEVICE)
    accuracy_pre = -1

    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(EPOCHS):
        #epoch +=1
        model.train()
        for batch_idx, (img, label) in enumerate(CAR_train):
            logits = model(img.to(DEVICE))
            loss = f_loss(logits, label.to(DEVICE))
            print('epoch :{}, batch :{}, loss :{:.4f}'.format(epoch, batch_idx, loss.sum().item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 2 == 0:
            model.eval()
            total_correct = 0
            total_num = 0
            with torch.no_grad():
                for img, label in CAR_test:
                    logits = model(img.to(DEVICE))
                    pred = logits.argmax(dim=1)
                    correct = torch.eq(pred, label.to(DEVICE)).float().sum().item()
                    total_correct += correct
                    total_num += img.size(0)
            accuracy = total_correct / total_num       #定义准确率
            print('epoch:', epoch, 'accuracy:', accuracy)

            if accuracy > accuracy_pre:
                accuracy_pre = accuracy
                ################################
                fd = open('gpu.dat', 'a+')
                fd.write('epoch {}'.format(epoch) + ': ' + str(accuracy) + '\n')
                fd.close()
                ################################
                save_path = os.path.join('./model', 'gpu_backup_' + str(epoch)+'.pth')
                torch.save(model.state_dict(), save_path)
    last_model = "./model/gpu_backup_last.pth"
    torch.save(model.state_dict(), last_model)


if __name__ == '__main__':
    main()
