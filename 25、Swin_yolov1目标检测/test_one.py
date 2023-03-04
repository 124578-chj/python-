from utils import encode_target,out_preds
import cv2
import numpy as np
from vgg_yolo import vgg19_bn
from dataloader import VOC_CLASSES
import torch
from transform import BaseTransform
from IPython import embed
import os
import time
def test_one_img(img_path):

    src = cv2.imread(img_path)
    h, w, _ = src.shape

    x=BaseTransform(img_size)(src)[0]
    x=x[:, :, (2, 1, 0)]
    x = torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0)
    net.eval()
    preds = net.forward(x) #out_preds(preds.detach())

    boxes, probs, cls_indexs = encode_target(preds,is_keep_nms=True)
    det = src.copy()

    if boxes.shape[0] >= 1:
        boxes1 = boxes.numpy()
        for i in range(boxes1.shape[0]):
            xmin, ymin, xmax, ymax = boxes1[i][0], boxes1[i][1], boxes1[i][2], boxes1[i][3]
            xmin *= w
            ymin *= h
            xmax *= w
            ymax *= h

            label_index = int(cls_indexs[i].item())
            cv2.rectangle(det, [int(xmin), int(ymin)], [int(xmax), int(ymax)], (0, 0, 255), 2)
            cv2.putText(det, str(VOC_CLASSES[(label_index)]), (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255))
            print('conf:{:.2f}'.format(probs[i].item()), 'label:', VOC_CLASSES[label_index])
    else:
        print('No object')

    cv2.imshow('detection', det)
    cv2.waitKey(0)

def test_video(video_path,use_cam=False,is_vis=False):
    if use_cam:
        print('Currently use camera....')
        cap=cv2.VideoCapture('http://admin:admin@10.41.132.46:8081/')
    else:
        cap = cv2.VideoCapture(video_path)
    fps = 0.0
    while(True):
        t1=time.time()
        ref,frame=cap.read()
        x = torch.from_numpy(BaseTransform(img_size)(frame)[0][:, :, (2, 1, 0)]).permute(2, 0, 1).unsqueeze(0)

        frame = cv2.resize(frame, (img_size, img_size))

        preds = net.forward(x)
        #是否对检测结果进行可视化
        if is_vis:
            boxes, probs, cls_indexs = encode_target(preds,is_keep_nms=True)
            det = frame.copy()
            if boxes.shape[0] >= 1:
                boxes1 = boxes.numpy()
                for i in range(boxes1.shape[0]):
                    xmin, ymin, xmax, ymax = boxes1[i][0], boxes1[i][1], boxes1[i][2], boxes1[i][3]
                    xmin *= img_size
                    ymin *= img_size
                    xmax *= img_size
                    ymax *= img_size

                    label_index = int(cls_indexs[i].item())
                    cv2.rectangle(det, [int(xmin), int(ymin)], [int(xmax), int(ymax)], (0, 0, 255), 2)
                    cv2.putText(det, str(VOC_CLASSES[(label_index)]), (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                (0, 0, 255))
                    print('conf:{:.2f}'.format(probs[i].item()), 'label:', VOC_CLASSES[label_index])
            cv2.imshow('Det',det)
        fps=(fps+(1./(time.time()-t1)))/2
        print('fps=%.2f'%fps)
        c=cv2.waitKey(10)
        if c==27:
            cap.release()
            break



if __name__=='__main__':
    model_weight=r'416_darknet19.pt'
    img_path = r'C:\Users\Administrator\Desktop\VOCdevkit\VOC2007\JPEGImages\000033.jpg'
    video_path=r'01.mp4'
    img_size=224
    num_class=len(VOC_CLASSES)
    net = vgg19_bn(pretrained=False, num_classes=num_class,image_size=img_size)
    if os.path.exists(model_weight):
        print('Succeeful Loading weight....')
        net.load_state_dict(torch.load(model_weight))

    #测试单张图片
    # test_one_img(img_path)
    #测试视频 or 摄像头
    test_video(video_path,use_cam=True,is_vis=True)