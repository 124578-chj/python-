import cv2
from vgg_yolo import vgg19_bn
from swin_yolo import *
from dataloader import Voc_data,VOC_CLASSES
from transform import SSDAugmentation,ColorAugmentation
import  torch.utils.data as data
import torch
from IPython import embed
from utils import nms,out_preds,encode_target
from visdom import Visdom
import numpy as np
from torch.autograd import Variable
from loss import YoloV1Loss
import os
import  matplotlib.pyplot as plt
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

def show_target_preds(img,target,preds):
    src=img[0].detach().cpu()
    boxes1,probs1,cls_indexs1=encode_target(preds[0].detach().cpu(),is_keep_nms=True) #preds
    src=src.permute(1,2,0).numpy().astype(np.uint8)
    det=src.copy()

    if boxes1.shape[0]>=1:
        boxes1 = boxes1.numpy()
        for i in range(boxes1.shape[0]):
            xmin, ymin, xmax, ymax = boxes1[i][0], boxes1[i][1], boxes1[i][2], boxes1[i][3]
            xmin *= img_size
            ymin *= img_size
            xmax *= img_size
            ymax *= img_size

            label_index = int(cls_indexs1[i].item())
            cv2.rectangle(det, [int(xmin), int(ymin)], [int(xmax), int(ymax)], (0, 0, 255), 2)
            cv2.putText(det, str(VOC_CLASSES[(label_index)]), (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255))
            print('conf:{:.2f}'.format(probs1[i].item()), 'label:', VOC_CLASSES[label_index])
    else:
        print('No object')

    cv2.imshow('Detection',det)
    cv2.waitKey(0)

def train_one(epoch):
    all_loss=AverageMeter()
    loss_loc=AverageMeter()
    loss_contain=AverageMeter()
    loss_no_contain=AverageMeter()
    loss_nooobj=AverageMeter()
    loss_class=AverageMeter()
    for batch_id, (img, target) in enumerate(data_train):
        img = Variable(img).cuda()
        target = Variable(target).cuda()
        preds=net.forward(img)
        # show_target_preds(img, target, preds)
        # from IPython import embed
        # embed()

        loc_loss,contain_loss,no_contain_loss,nooobj_loss,class_loss=cal_loss(preds,target)
        total_loss=loc_loss+contain_loss+no_contain_loss+nooobj_loss+class_loss

        optimizer.zero_grad()

        total_loss.backward()
        optimizer.step()

        all_loss.update(total_loss.item());loss_loc.update(loc_loss.item())
        loss_contain.update(contain_loss.item());loss_no_contain.update(no_contain_loss.item())
        loss_nooobj.update(nooobj_loss.item());loss_class.update(class_loss.item())

        if batch_id %5==0:
            print('Batch/Batchs:[{}/{}],'.format(batch_id, len(data_train)),
                  'epoch/epochs:[{}/{}],'.format(epoch, epochs),
                  'Loss:{:.3f}'.format(total_loss.item())
                  )
    Y=np.column_stack([all_loss.avg, loss_loc.avg, loss_contain.avg, loss_no_contain.avg,loss_nooobj.avg,loss_class.avg])
    X= np.column_stack([epoch, epoch, epoch, epoch,epoch,epoch])
    vit.line(Y,X, update='append', win='train',
             opts=dict(showlegend=True, linecolor=np.array(
                 [[0,0,255],
                  [125, 0, 169],
                  [0, 25, 98],
                  [0, 125, 169],
                  [125, 26, 169],
                  [39, 125, 36]]),
                       title="train loss",
                       legend=['all_loss', 'loss_loc', 'loss_contain', 'loss_no_contain','loss_nooobj','loss_class'],
                       dash=np.array(['solid', 'solid', 'solid', 'solid', 'solid', 'solid']),
                       xlabel='epoch'))

if __name__=='__main__':
    num_class=20
    img_size = 224
    batch_size=64
    epochs=601
    learning_rate=0.01
    model_weight = r'swin_224_tiny.pt'
    data_path = r'C:\Users\Administrator\Desktop\VOCdevkit\VOC2007'
    train_data = Voc_data(data_path,name='train',transform=SSDAugmentation(img_size))
    data_train = data.DataLoader(train_data, batch_size=batch_size,drop_last=True)
    #定义训练网络
    net = swin_tiny_patch4_window7_224(num_classes=num_class)

    # net = vgg19_bn(pretrained=True, num_classes=num_class,image_size=img_size)  # 输出预测值 [B,7,7,2x5+num_class] 5=[x,y,w,h,conf]


    net.cuda()
    if os.path.exists(model_weight):
        print('Succeeful Loading weight....')
        net.load_state_dict(torch.load(model_weight))
    #定义损失
    cal_loss = YoloV1Loss()
    #定义Visdom
    vit=Visdom(port=6006)

    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)



    for epoch in range(epochs):
        train_one(epoch)
        torch.save(net.state_dict(),'swin_224_tiny.pt')








