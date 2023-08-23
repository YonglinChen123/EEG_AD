# -*- coding: utf-8 -*-
import os, sys, glob, argparse
import numpy as np
import mne
import time, datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset

from resnet import Resnet18
from ViTnet import ViT

train_jpg = np.array(glob.glob('./Data3/*/*.set'))
###########################################################

def dataprosss1(raw):
    sfreq = raw.info['sfreq']
    data1, times1 = raw[0:19, int(sfreq * 0):int(sfreq * 4)]
    data2, times2 = raw[0:19, int(sfreq * 4):int(sfreq * 8)]
    data3, times3 = raw[0:19, int(sfreq * 8):int(sfreq * 12)]
    data4, times4 = raw[0:19, int(sfreq * 12):int(sfreq * 16)]
    data5, times5 = raw[0:19, int(sfreq * 16):int(sfreq * 20)]
    data6, times6 = raw[0:19, int(sfreq * 20):int(sfreq * 24)]
    data7, times7 = raw[0:19, int(sfreq * 24):int(sfreq * 28)]
    data8, times8 = raw[0:19, int(sfreq * 28):int(sfreq * 32)]
    data9, times9 = raw[0:19, int(sfreq * 32):int(sfreq * 36)]
    data10, times10 = raw[0:19, int(sfreq * 36):int(sfreq * 40)]


    data1 = torch.Tensor(data1).float()
    data1 = data1.unsqueeze(0)
    data2 = torch.Tensor(data2).float()
    data2 = data2.unsqueeze(0)
    data2 = torch.cat((data1, data2), 0)
    data3 = torch.Tensor(data3).float()
    data3 = data3.unsqueeze(0)
    data3 = torch.cat((data3, data2), 0)
    data4 = torch.Tensor(data4).float()
    data4 = data4.unsqueeze(0)
    data4 = torch.cat((data4, data3), 0)
    data5 = torch.Tensor(data5).float()
    data5 = data5.unsqueeze(0)
    data5 = torch.cat((data5, data4), 0)
    data6 = torch.Tensor(data6).float()
    data6 = data6.unsqueeze(0)
    data6 = torch.cat((data6, data5), 0)
    data7 = torch.Tensor(data7).float()
    data7 = data7.unsqueeze(0)
    data7 = torch.cat((data7, data6), 0)
    data8 = torch.Tensor(data8).float()
    data8 = data8.unsqueeze(0)
    data8 = torch.cat((data8, data7), 0)
    data9 = torch.Tensor(data9).float()
    data9 = data9.unsqueeze(0)
    data9 = torch.cat((data9, data8), 0)
    data10 = torch.Tensor(data10).float()
    data10 = data10.unsqueeze(0)
    data10 = torch.cat((data10, data9), 0)

    data251 = torch.cat((data10, data10), 0)
    data251 = torch.cat((data251, data10), 0)
    resize = transforms.Resize([32, 32])
    data251 = resize(data251)

########################################################
    data1_1 = raw.compute_psd(method='welch', fmin=0.5, fmax=4)
    data1_1 = data1_1.get_data()
    data1_1 = torch.Tensor(data1_1).float()
    data1_1 = data1_1.unsqueeze(0)
    data1_2 = torch.cat((data1_1, data1_1), 0)
    data1_2 = torch.cat((data1_1, data1_2), 0)


    data2_1 = raw.compute_psd(method='welch', fmin=4, fmax=8)
    data2_1 = data2_1.get_data()
    data2_1 = torch.Tensor(data2_1).float()
    data2_1 = data2_1.unsqueeze(0)
    data2_2 = torch.cat((data2_1, data2_1), 0)
    data2_2 = torch.cat((data2_1, data2_2), 0)

    data3_1 = raw.compute_psd(method='welch', fmin=8, fmax=13)
    data3_1 = data3_1.get_data()
    data3_1 = torch.Tensor(data3_1).float()
    data3_1 = data3_1.unsqueeze(0)
    data3_2 = torch.cat((data3_1, data3_1), 0)
    data3_2 = torch.cat((data3_1, data3_2), 0)

    data4_1 = raw.compute_psd(method='welch', fmin=13, fmax=25)
    data4_1 = data4_1.get_data()
    data4_1 = torch.Tensor(data4_1).float()
    data4_1 = data4_1.unsqueeze(0)
    data4_2 = torch.cat((data4_1, data4_1), 0)
    data4_2 = torch.cat((data4_1, data4_2), 0)

    data5_1 = raw.compute_psd(method='welch', fmin=25, fmax=45)
    data5_1 = data5_1.get_data()
    data5_1 = torch.Tensor(data5_1).float()
    data5_1 = data5_1.unsqueeze(0)
    data5_2 = torch.cat((data5_1, data5_1), 0)
    data5_2 = torch.cat((data5_1, data5_2), 0)


########################################################
    #list_train = torch.Tensor(list_train).float()
    return data251, data1_2, data2_2, data3_2, data4_2, data5_2
#######################################################
class QRDataset(Dataset):
    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        #start_time = time.time()
        raw = mne.io.read_raw_eeglab(self.train_jpg[index], preload=True)
        img1, img2_1, img2_2, img2_3, img2_4, img2_5 = dataprosss1(raw)

        label1 = 0
        label2 = 0
        if 'NC' in self.train_jpg[index]:
            label1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            label2 = 0
        elif 'AD' in self.train_jpg[index]:
            label1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            label2 = 1
        elif 'MCI' in self.train_jpg[index]:
            label1 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            label2 = 2
        #label1 = torch.from_numpy(np.array(label1))
############################################################
############################################################
        return img1, img2_1, img2_2, img2_3, img2_4, img2_5, torch.from_numpy(np.array(label1)), torch.from_numpy(np.array(label2))

    def __len__(self):
        return len(self.train_jpg)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""

    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

# -----------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 4, common_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

class VisitNet(nn.Module):
    def __init__(self):
        super(VisitNet, self).__init__()
        model1 = Resnet18(3)
        self.resnet1 = model1
    def forward(self, img):
        out = self.resnet1(img)
        return out


def validate(val_loader, model, model2,criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@2', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input1, input2, input3, input4, input5, input6, target1, target2) in enumerate(val_loader):
            input1 = input1.cuda(non_blocking=True)
            input2 = input2.cuda(non_blocking=True)
            input3 = input3.cuda(non_blocking=True)
            input4 = input4.cuda(non_blocking=True)
            input5 = input5.cuda(non_blocking=True)
            input6 = input6.cuda(non_blocking=True)
            target1 = target1.cuda(non_blocking=True)
            target2 = target2.cuda(non_blocking=True)

            # compute output
            input1 = input1.reshape([-1, 3, 32, 32])


            output1 = model2(input1)
            output2 = model(input2)
            output3 = model(input3)
            output4 = model(input4)
            output5 = model(input5)
            output6 = model(input6)
            output = torch.cat((output1, output2), 0)
            output = torch.cat((output, output3), 0)
            output = torch.cat((output, output4), 0)
            output = torch.cat((output, output5), 0)
            output = torch.cat((output, output6), 0)
            # target2 = target.long()
            target1 = target1.reshape([-1])
            target = torch.cat((target1, target2), 0)
            target = torch.cat((target, target2), 0)
            target = torch.cat((target, target2), 0)
            target = torch.cat((target, target2), 0)
            target = torch.cat((target, target2), 0)

            loss = criterion(output, target.long())
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), 240)
            top1.update(acc1[0], 240)
            top5.update(acc5[0], 240)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        # print('Acc@5 {top5.avg:.3f}'
        #       .format(top5=top5))
        return top1


def predict(test_loader, model, tta=10):
    # switch to evaluate mode
    model.eval()

    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            end = time.time()
            for i, (input1, input2, input3, input4, input5, input6, target1, target2) in enumerate(test_loader):

                # compute output
                #output = model(input, path)
                output = output.data.cpu().numpy()

                test_pred.append(output)
        test_pred = np.vstack(test_pred)

        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta


def train(train_loader, model, model2, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)
    model.train()
    model2.train()
    end = time.time()
    for i, (input1, input2, input3, input4, input5, input6, target1, target2) in enumerate(train_loader):
        input1 = input1.cuda(non_blocking=True)
        input2 = input2.cuda(non_blocking=True)
        input3 = input3.cuda(non_blocking=True)
        input4 = input4.cuda(non_blocking=True)
        input5 = input5.cuda(non_blocking=True)
        input6 = input6.cuda(non_blocking=True)
        target1 = target1.cuda(non_blocking=True)
        target2 = target2.cuda(non_blocking=True)
        input1= input1.reshape([-1, 3, 32,32])
        output1 = model2(input1)
        output2 = model(input2)
        output3 = model(input3)
        output4 = model(input4)
        output5 = model(input5)
        output6 = model(input6)
        output = torch.cat((output1, output2), 0)
        output = torch.cat((output, output3), 0)
        output = torch.cat((output, output4), 0)
        output = torch.cat((output, output5), 0)
        output = torch.cat((output, output6), 0)
        target1 = target1.reshape([-1])
        target = torch.cat((target1, target2), 0)
        target = torch.cat((target, target2), 0)
        target = torch.cat((target, target2), 0)
        target = torch.cat((target, target2), 0)
        target = torch.cat((target, target2), 0)

        loss = criterion(output, target.long())
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), 240)
        top1.update(acc1[0], 240)
        top5.update(acc5[0], 240)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.pr2int(i)



skf = KFold(n_splits=10, random_state=233, shuffle=True)
for flod_idx, (train_idx, val_idx) in enumerate(skf.split(train_jpg, train_jpg)):

    train_loader = torch.utils.data.DataLoader(
        QRDataset(train_jpg[train_idx] ), batch_size=8, shuffle=True, num_workers=0, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        QRDataset(train_jpg[val_idx]
                  ), batch_size=8, shuffle=False, num_workers=0, pin_memory=True
    )

    model = VisitNet().cuda()
    model2 = ViT(
        image_size=32,
        patch_size=16,
        num_classes=3,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    ).cuda()


    criterion = nn.CrossEntropyLoss().cuda()
    #1
    optimizer = torch.optim.SGD(model.parameters(), 0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
    best_acc = 0.0
    for epoch in range(200):
        scheduler.step()
        print('Epoch: ', epoch)

        train(train_loader, model, model2, criterion, optimizer,epoch)
        val_acc = validate(val_loader, model, model2, criterion)

        if val_acc.avg.item() > best_acc:
            best_acc = val_acc.avg.item()
            torch.save(model.state_dict(), './3_fold{0}.pt'.format(flod_idx))

    break