import shutil
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torchvision import transforms
from torch.utils.data import DataLoader
from loss import *
from dataloader import *
import lanenet
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', default='')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--new_length', default=705, type=int)
parser.add_argument('--new_width', default=833, type=int)
parser.add_argument('--label_length', default=177, type=int)
parser.add_argument('--label_width', default=209, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 200)')
parser.add_argument('--resume', default='checkpoints', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec = 0


def main():
    global args, best_prec
    args = parser.parse_args()
    print ("Build model ...")
    model = lanenet.Net()
    model = torch.nn.DataParallel(model).cuda()
    #model.apply(weights_init)
    #params = torch.load('checkpoints/old.pth.tar')
    #model.load_state_dict(params['state_dict'])
    if not os.path.exists(args.resume):
        os.makedirs(args.resume)
    print("Saving everything to directory %s." % (args.resume))

    # define loss function (criterion) and optimizer
    criterion = cross_entropy2d
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    cudnn.benchmark = True

    # data transform
    
    train_data = MyDataset('/mnt/lustre/share/dingmingyu/new_list_lane.txt', args.dir_path, args.new_width, args.new_length,args.label_width,args.label_length)

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,pin_memory=True)


    for epoch in range(args.start_epoch, args.epochs):
        print 'epoch: ' + str(epoch + 1)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set

        # remember best prec and save checkpoint

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_name = "%03d_%s" % (epoch + 1, "checkpoint.pth.tar")
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, checkpoint_name, args.resume)



def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_seg = AverageMeter()
    losses_ins = AverageMeter()
    losses_cls = AverageMeter()
    losses_hnet = AverageMeter()
    losses_width = AverageMeter()
    lrs = AverageMeter()
    # switch to train mode
    model.train()
    weight_cus = torch.ones(2)
    weight_cus[1] = 2
    weight_cus = weight_cus.cuda()
    end = time.time()
    for i, (input, target_ins, n_objects, thining_gt) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr = adjust_learning_rate(optimizer, epoch*len(train_loader)+i, args.epochs*len(train_loader))
        lrs.update(lr)
        input = input.float().cuda()
        target01 = target_ins.sum(1)
        target01 = target01.long().cuda()


        target_ins = target_ins.float().cuda()
        #n_objects = n_objects.long().cuda()

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target01)
        target_ins_var = torch.autograd.Variable(target_ins)
        #n_objects_var = torch.autograd.Variable(n_objects)

        n_objects_normalized = n_objects.float().cuda() / 4
        n_objects_normalized_var = torch.autograd.Variable(n_objects_normalized)
        x_sem, x_ins, x_cls, x_hnet= model(input_var)
        criterion_mse = torch.nn.MSELoss().cuda()
        #x_hnet = x_hnet * 10
        
        
        ##############################################  HNET  ##################################################
        
        n, c = x_hnet.size()
        thining_gt = thining_gt.numpy()
        loss_hnet = torch.autograd.Variable(torch.zeros(1).float().cuda())
        avg_size = 0
        loss_width = torch.autograd.Variable(torch.zeros(1).float().cuda())
        avg_width = 0
        #print x_hnet
        for n_index in range(n):
            H = torch.autograd.Variable(torch.zeros((3,3)).float().cuda())
            H[0,0] = x_hnet[n_index,0]
            H[0,1] = x_hnet[n_index,1]
            H[0,2] = x_hnet[n_index,2]
            H[1,1] = x_hnet[n_index,3]
            H[1,2] = x_hnet[n_index,4]
            H[2,1] = x_hnet[n_index,5]
            H[2,2] = 1
            if (thining_gt[n_index] > 0).any():
                bird_list = []
                for index in range(1, 5):
                    list_y, list_x = np.where(thining_gt[n_index] == index)
                    if len(list_y):
                        avg_size += 1
                        old_x = torch.autograd.Variable(torch.from_numpy(list_x).float().cuda())
                        p = np.vstack((list_x, list_y, np.ones(len(list_y)))).astype(np.float64)
                        P = torch.autograd.Variable(torch.from_numpy(p).float().cuda())
                        P_trans = torch.mm(H,P)
                        list_x = P_trans[0]
                        list_y = P_trans[1]
                        scale = P_trans[2]
                        Y = torch.stack((list_y**2, list_y, torch.autograd.Variable(torch.ones(list_y.size()[0]).float().cuda())), dim=0)
                        W = torch.inverse(Y.mm(Y.t()))#
                        W = W.mm(Y).mm(list_x.unsqueeze(1)) 
                        new_x = W[0]*list_y**2 + W[1]*list_y + W[2]
                        new_P = torch.stack((new_x, list_y, scale), dim=0)
                        trans_x = torch.inverse(H).mm(new_P)[0]
                        if index==2 or index==3:
                            bird_list.append((W,list_y))
                        if criterion_mse(trans_x/50, old_x/50).data.cpu().numpy() < 1:
                            loss_hnet += criterion_mse(trans_x/50, old_x/50)
                if len(bird_list) == 2:
                    miny = torch.min(bird_list[0][1].min(), bird_list[1][1].min())
                    maxy = torch.max(bird_list[0][1].max(), bird_list[1][1].max())
                    width_list = []
                    for yy in torch.arange(float(miny.data.cpu().numpy()),float(maxy.data.cpu().numpy()),float(((maxy-miny)/10).data.cpu().numpy())):
                        a = bird_list[0][0][0]*yy**2 + bird_list[0][0][1]*yy + bird_list[0][0][2]
                        b = bird_list[1][0][0]*yy**2 + bird_list[1][0][1]*yy + bird_list[1][0][2]
                        width_list.append(a - b)
                    if len(width_list)>= 10:
                        avg_width += 1
                        width_variable = torch.cat((width_list[0],width_list[1]),0)
                        for xx in range(2,10):
                            width_variable = torch.cat((width_variable,width_list[xx]),0)
                        width_variable = width_variable/width_variable.mean()
                        mean_variable = torch.autograd.Variable(torch.ones(10).float().cuda())
                        loss_width += criterion_mse(width_variable, mean_variable)/3
                        #else:
                        #    print trans_x, old_x
        if avg_width:
            loss_width /= avg_width
        loss_hnet /= avg_size               
        loss_cls = criterion_mse(x_cls, n_objects_normalized_var)
        #print x_sem.size(), x_ins.size(), target_ins_var.size(), n_objects_var.size() (256L, 2L, 177L, 209L) (256L, 4L, 177L, 209L) (256L, 4L, 177L, 209L) (256L,)   
        loss_ins = discriminative_loss(x_ins, target_ins_var, n_objects, 4, usegpu=True)
        #print x_sem.size(),target_var.size()
        loss_seg = criterion(x_sem, target_var, weight= weight_cus, size_average=True)
        loss = loss_seg + loss_ins + loss_cls + loss_hnet + loss_width

        losses_seg.update(loss_seg.data[0], input.size(0))
        losses_ins.update(loss_ins.data[0], input.size(0))
        losses_cls.update(loss_cls.data[0], input.size(0))
        losses_hnet.update(loss_hnet.data[0], input.size(0))
        losses_width.update(loss_width.data[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_seg {loss_seg.val:.4f} ({loss_seg.avg:.4f})\t'
                  'Loss_ins {loss_ins.val:.4f} ({loss_ins.avg:.4f})\t'
                  'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                  'Loss_hnet {loss_hnet.val:.4f} ({loss_hnet.avg:.4f})\t'
                  'Loss_width {loss_width.val:.4f} ({loss_width.avg:.4f})\t'
                  'Lr {lr.val:.5f} ({lr.avg:.5f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_seg=losses_seg, loss_ins=losses_ins, loss_cls=losses_cls, loss_hnet=losses_hnet, loss_width=losses_width, lr=lrs))



def save_checkpoint(state, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, curr_iter, max_iter, power=0.9):
    lr = args.lr * (1 - float(curr_iter)/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

if __name__ == '__main__':
    main()
