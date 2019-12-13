import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from model import PSPNet
from loss import PSPLoss
from data import make_datapath_list, DataTransform, VOCDataset


def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs):
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(device)

    net.to(device)
    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloaders_dict['train'].dataset)
    num_val_imgs = len(dataloaders_dict['val'].dataset)

    iteration = 1
    logs = []

    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                print('(train)')
            else:
                net.eval()
                print('(eval)')

            # minibatchの処理
            for imgs, anno_class_imgs in dataloaders_dict[phase]:
                imgs = imgs.to(device)
                anno_class_imgs = anno_class_imgs.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(imgs)
                    loss = criterion(outputs, anno_class_imgs.long())
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if iteration % 10 == 0:
                            print('Iteration: {} | Loss: {:.4f}'.format(iteration, loss.item()))
                        epoch_train_loss += loss.item()

                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()

        scheduler.step()

        print('Epoch: {} | Epoch_train_loss: {:.4f} | Epoch_val_loss: {:.4f}'.format(
            epoch + 1,
            epoch_train_loss / len(dataloaders_dict['train']),
            epoch_val_loss / len(dataloaders_dict['val'])))

        # logging
        log_epoch = {'epoch': epoch + 1,
                     'train_loss': epoch_train_loss / num_train_imgs,
                     'val_loss': epoch_val_loss / num_val_imgs}
        logs.append(log_epoch)

    torch.save(net.state_dict(), 'weights/pspnet50_' + str(epoch + 1) + '.pth')

    df = pd.DataFrame(logs)
    df.to_csv('log_output.csv')


def main():
    # Datasetの作成o
    rootpath = './data/VOCdevkit/VOC2012/'
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath=rootpath)

    color_mean = (0.485, 0.456, 0.406)
    color_std = (0.229, 0.224, 0.225)

    train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train',
                               transform=DataTransform(475, color_mean, color_std))
    val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val',
                             transform=DataTransform(475, color_mean, color_std))

    # DataLoaderの作成
    batch_size = 4

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}

    # ADE20Kで訓練した学習済みモデルをベースにFine-tuningする
    net = PSPNet(n_classes=150)
    state_dict = torch.load('./weights/pspnet50_ADE20K.pth')
    net.load_state_dict(state_dict)

    # VOC2012データでFine-tuningするためヘッドの部分だけ付け替え
    n_classes = 21
    net.decode_feature.classification = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
    net.aux.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    net.decode_feature.classification.apply(init_weights)
    net.aux.classification.apply(init_weights)
    print(net)

    criterion = PSPLoss(aux_weight=0.4)

    optimizer = optim.SGD([
        {'params': net.feature_conv.parameters(), 'lr': 1e-3},
        {'params': net.feature_res_1.parameters(), 'lr': 1e-3},
        {'params': net.feature_res_2.parameters(), 'lr': 1e-3},
        {'params': net.feature_dilated_res_1.parameters(), 'lr': 1e-3},
        {'params': net.feature_dilated_res_2.parameters(), 'lr': 1e-3},
        {'params': net.pyramid_pooling.parameters(), 'lr': 1e-3},
        {'params': net.decode_feature.parameters(), 'lr': 1e-2},
        {'params': net.aux.parameters(), 'lr': 1e-2}
    ], momentum=0.9, weight_decay=0.0001)

    # scheduler
    def lambda_epoch(epoch):
        max_epoch = 30
        return math.pow((1 - epoch / max_epoch), 0.9)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

    num_epochs = 30
    train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs=num_epochs)


if __name__ == "__main__":
    main()
