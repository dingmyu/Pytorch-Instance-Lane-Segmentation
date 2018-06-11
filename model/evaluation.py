import shutil
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import lanenet
import os
import torchvision as tv
from torch.autograd import Variable
from PIL import Image
import cv2
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth

parser = argparse.ArgumentParser()
parser.add_argument('--img_list', dest='img_list', default='/mnt/lustre/sunpeng/ADAS_lane_prob_IoU_evaluation/GMXL/new_camera1212_test.txt',
                            help='the test image list', type=str)

parser.add_argument('--img_dir', dest='img_dir', default='/mnt/lustre/drive_data/adas-video/record/',
                        help='the test image dir', type=str)
parser.add_argument('--gtpng_dir', dest='gtpng_dir', default='/mnt/lustre/drive_data/adas-video/record/',
                        help='the test png label dir', type=str)

parser.add_argument('--model_path', dest='model_path', default='checkpoints/020_checkpoint.pth.tar',
                        help='the test model', type=str)

parser.add_argument('--prethreshold', dest='prethreshold', default=0.4,
                        help='preset bad threshold', type=float)


def get_iou(gt, prob):
    cross = np.multiply(gt, prob)
    unite = gt + prob
    unite[unite >= 1] = 1
    union = np.sum(unite)
    inter = np.sum(cross)
    if (union != 0):
        iou = inter * 1.0 / union
        if (np.sum(gt) == 0):
            iou = -1
    else:
        iou = -10
    return iou

def iou_one_frame(gtpng, rprob, bias_threshold):
    iou = [0, 0, 0, 0] 
    gt = [gtpng.copy(),gtpng.copy(),gtpng.copy(),gtpng.copy()]
    for i, item in enumerate(gt):
        item[item != i+1] = 0
        item[item == i+1] = 1
    prob = [rprob.copy(),rprob.copy(),rprob.copy(),rprob.copy()]
    for i, item in enumerate(prob):
        item[item != i+1] = 0
        item[item == i+1] = 1
    iou_all = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            iou_all[i,j] = get_iou(gt[i],prob[j])
    #print iou_all
    for i in range(4):
        max = np.argmax(iou_all)
        gt_max = int(max/4)
        prob_max = max % 4
        #print gt_max, prob_max
        iou[gt_max] = iou_all.max()
        iou_all[gt_max,:] = -100
        iou_all[:,prob_max] = -100
    return iou



def main():
    global args
    args = parser.parse_args()
    print ("Build model ...")
    model = lanenet.Net()
    model = torch.nn.DataParallel(model).cuda()
    state = torch.load(args.model_path)['state_dict']
    model.load_state_dict(state)
    model.eval()    

    mean = [104, 117, 123]
    f = open(args.img_list)
    ni = 0
    count_gt = [0,0,0,0]
    total_iou = [0,0,0,0]
    total_iou_big = [0,0,0,0]
    for line in f:
        #if ni>1:
        #    break
        line = line.strip()
        arrl = line.split(" ")
    
        #gtlb = cv2.imread(args.gtpng_dir + arrl[1], -1)
        gtlb = cv2.imread(args.gtpng_dir + arrl[1], -1)
        #print gtlb.shape
        gt_num_list = list(np.unique(gtlb))
        gt_num_list.remove(0)
        n_objects_gt = len(gt_num_list)
        image = cv2.imread(args.img_dir + arrl[0])
        image = cv2.resize(image,(833,705)).astype(np.float32)
        image -= mean
        image = image.transpose(2, 0, 1)
        #image = cv2.imread(args.img_dir + arrl[0], -1)
        #image = cv2.resize(image, (732,704), interpolation = cv2.INTER_NEAREST)
        #print image.shape
        image = torch.from_numpy(image).unsqueeze(0)
        image = Variable(image.float().cuda(0), volatile=True)
        output, embedding, n_objects_predictions = model(image)
        output = torch.nn.functional.softmax(output[0],dim=0)
        prob = output.data.cpu().numpy()
        embedding = embedding.data.cpu().numpy()
        n_objects_predictions = n_objects_predictions * 4
        n_objects_predictions = torch.round(n_objects_predictions).int()
        n_objects_predictions = int(n_objects_predictions.data.cpu())
        print n_objects_predictions,'~~~~~~~~~~~~~~~~~~~~~~~`'

        if not n_objects_predictions:
            continue
        prob[prob >= args.prethreshold] = 1.0
        prob[prob < args.prethreshold] = 0
        embedding = embedding[0,:,:,:].transpose((1, 2, 0))
        #print prob.shape
        mylist = []
        indexlist = []
        for i in range(embedding.shape[0]):
            for j in range(embedding.shape[1]):
                if prob[1][i][j] > 0:
                    mylist.append(embedding[i,j,:])
                    indexlist.append((i,j))
        if not mylist:
            continue
        mylist = np.array(mylist)
       # bandwidth = estimate_bandwidth(mylist, quantile=0.3, n_samples=100, n_jobs = 8)
       # print bandwidth
       # estimator = MeanShift(bandwidth=1, bin_seeding=True)
        estimator = KMeans(n_clusters = n_objects_predictions)
        #estimator = AffinityPropagation(preference=-0.4, damping = 0.5)
        t = time.time()
        estimator.fit(mylist)
        print time.time() - t
        for i in range(4):
            print len(estimator.labels_[estimator.labels_==i])
        #print len(np.unique(estimator.labels_)),'~~~~~~~~~~~~~~~~'
        new_prob = np.zeros((embedding.shape[0],embedding.shape[1]),dtype=int)
        for index, item in enumerate(estimator.labels_):
            if item <= 4:
                new_prob[indexlist[index][0]][indexlist[index][1]] = item + 1

        gtlb = cv2.resize(gtlb, (prob.shape[2], prob.shape[1]),interpolation = cv2.INTER_NEAREST)
        iou = iou_one_frame(gtlb, new_prob, args.prethreshold)

        print('IoU of ' + str(ni) + ' '+ arrl[0] + ': ' + str(iou))
        for i in range(0,4):
            if iou[i] >= 0:
                count_gt[i] = count_gt[i] + 1
                total_iou[i] = total_iou[i] + iou[i]
        ni += 1
    mean_iou = np.divide(total_iou, count_gt)
    print('Image numer: ' + str(ni))
    print('Mean IoU of four lanes: ' + str(mean_iou))
    print('Overall evaluation: ' + str(mean_iou[0] * 0.2 + mean_iou[1] * 0.3 + mean_iou[2] * 0.3 + mean_iou[3] * 0.2))

    f.close()

if __name__ == '__main__':
    main()

