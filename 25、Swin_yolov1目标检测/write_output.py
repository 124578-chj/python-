from utils import encode_target,out_preds
import cv2
import numpy as np
from vgg_yolo import vgg19_bn
from torch.autograd import Variable
from dataloader import VOC_CLASSES
import torch
from transform import BaseTransform
from IPython import embed
import os
from dataloader import Voc_data,VOC_CLASSES
from transform import SSDAugmentation,ColorAugmentation
import  torch.utils.data as data


def write_txt(name,bbox,probs,label,is_target):
    if is_target:
        file_txt = os.getcwd() + '/output/ground-truth/' + name + '.txt'
    else:
        file_txt=os.getcwd()+'/output/detection-results/'+ name + '.txt'
    file=open(file_txt,'w')
    for i in range(bbox.shape[0]):
        xmin, ymin, xmax, ymax = bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]
        xmin *= img_size
        ymin *= img_size
        xmax *= img_size
        ymax *= img_size

        label_index = int(label[i].item())

        if is_target:
            file.writelines(VOC_CLASSES[label_index] +' '
                            + str(int(xmin.item())) + ' ' + str(int(ymin.item())) + ' ' + str(
                int(xmax.item())) + ' ' + str(int(ymax.item())))
        else:
            file.writelines(VOC_CLASSES[label_index]+' '+str(round(probs[i].item(),6))+' '
                            +str(int(xmin.item()))+' '+str(int(ymin.item()))+' '+str(int(xmax.item()))+' '+str(int(ymax.item())))
        file.writelines('\n')




if __name__=='__main__':
    data_path = r'C:\Users\Administrator\Desktop\VOCdevkit\VOC2007'
    img_path = data_path + '\JPEGImages\\'
    model_weight=r'416_darknet19.pt'
    file_name=['detection-results','ground-truth','images-optional']
    img_size=224
    num_class=len(VOC_CLASSES)
    net = vgg19_bn(pretrained=False, num_classes=num_class,image_size=img_size);net.cuda()
    if os.path.exists(model_weight):
        print('Succeeful Loading weight....')
        net.load_state_dict(torch.load(model_weight))

    #如果目标路径不存在，新建这个文件夹
    for file_path in file_name:
        if not os.path.exists(os.getcwd()+r'/output/'+file_path):
            os.makedirs(os.getcwd()+r'/output/'+file_path)

    detection=os.getcwd()+r'/output/detection-results'
    gt=os.getcwd()+r'/output/ground-truth'
    img_out=os.getcwd()+r'/output/images-optional'
    val_data = Voc_data(data_path, name='val', transform=BaseTransform(img_size))
    data_val = data.DataLoader(val_data, batch_size=1, drop_last=True)
    net.eval()


    for batch_id, (img, target) in enumerate(data_val):
        img = Variable(img).cuda()
        target = Variable(target).cuda()
        preds=net.forward(img)
        p_boxes, p_probs, p_cls_indexs = encode_target(preds[0].detach().cpu(), is_keep_nms=True)
        t_boxes, t_probs, t_cls_indexs = encode_target(target[0].detach().cpu(), is_keep_nms=True)
        img_name=val_data.train_id[batch_id]

        write_txt(img_name,p_boxes,p_probs, p_cls_indexs,is_target=False)
        write_txt(img_name, t_boxes, t_probs, t_cls_indexs,is_target=True)

    for ids, name in enumerate(val_data.train_id):
        src = cv2.imread(img_path + name + '.jpg')
        src=cv2.resize(src,(img_size,img_size))
        cv2.imwrite(img_out + '/' + name + '.jpg', src)
        # cv2.imshow('img',src)
        # cv2.waitKey(0)




