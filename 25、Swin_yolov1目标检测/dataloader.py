import torch.utils.data as data
import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import transforms
from IPython import embed
import cv2

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
'sheep', 'sofa', 'train', 'tvmonitor']

class Voc_data(data.Dataset):
    def __init__(self,data_path,name,transform=None):

        self.txt_path=data_path+'\ImageSets\Main\\{}.txt'.format(name)
        self.img_path=data_path+'\JPEGImages\\'
        self.label_path=data_path+'\Annotations\\'
        self.B = 2  #单个格子产生预测框
        self.num_class = 20  # 总的类别数量
        self.grid_num = 7 #图片每一行每一列划分多少格子
        self.train_id = []
        self.transform=transform
        self.get_data()
    def __getitem__(self, item):
        img,target=self.pull_item(ids=item)

        target=self.decode_targt(target)  #[x,y,w,h]
        return img,torch.tensor(target)

    def __len__(self):
        return len(self.train_id)

    def get_data(self):
        if not os.path.exists(self.txt_path):
            print('Train file txt path error...')
            raise FileNotFoundError
        with open(self.txt_path,'r') as f:
            for line in f.readlines():
                self.train_id.append(line.strip('\n'))

    def pull_item(self,ids):
        img = cv2.imread(self.img_path + self.train_id[ids] + '.jpg')#Image.open()
        h,w,_=img.shape
        file_id=self.train_id[ids]
        k=self.label_path+file_id+'.xml'
        tree=ET.parse(k)
        root=tree.getroot()
        target=list()
        for bbox in root.iter('object'):
            # print('class_name:',VOC_CLASSES.index(bbox[0].text))
            temp=[]
            for b in bbox.find('bndbox'):
                temp.append(float(b.text))
                # print(b.tag,b.text)
            temp.append(VOC_CLASSES.index(bbox[0].text))
            target.append(temp)

        if len(target) == 0:
            target = np.zeros([1, 5])
        else:
            target = np.array(target)

        if self.transform is not None:
            target[:, :4]/=np.array([w,h,w,h])
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # to tensor
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            # target
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, target #[xmin,ymin,xmax,ymax,labels]

    def decode_targt(self,bbox):
        """
        :param target: [[xmin,ymin,xmax,ymax,label],....]
        :return: 7x7x[2x5+20]
        """

        target = np.zeros((self.grid_num, self.grid_num, (self.B*5+self.num_class)),dtype=np.float32)  # 7x7x(5xB+C) B:[x1,y1,x2,y2,置信度]
        cell_size = 1. / self.grid_num
        wh = bbox[:, 2:4] - bbox[:, :2]  # 标签框的长宽
        cxcy = (bbox[:, 2:4] + bbox[:, :2]) / 2  # 小格子的中心
        labels=bbox[:,4:].squeeze(-1)


        for i in range(cxcy.shape[0]):
            cxcy_sample = cxcy[i]
            ij = np.ceil(cxcy_sample / cell_size) - 1  #
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 10] = 1
            xy = ij * cell_size  # 匹配到的网格的左上角相对坐标

            delta_xy = (cxcy_sample - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return target



if __name__=='__main__':
    img_size=600
    data_path=r'C:\Users\Administrator\Desktop\VOCdevkit\VOC2007'


    def base_transform(image, size, mean):
        x = cv2.resize(image, (size, size)).astype(np.float32)
        x -= mean
        x = x.astype(np.float32)
        return x

    class BaseTransform:
        def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
            self.size = size
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)

        def __call__(self, image, boxes=None, labels=None):
            return base_transform(image, self.size, self.mean), boxes, labels

    from transform import SSDAugmentation,ColorAugmentation

    train_data=Voc_data(data_path,name='val',transform=BaseTransform(img_size))
    # data_train=data.DataLoader(train_data,batch_size=2)

    for i in range(1000):
        im, gt = train_data.pull_item(i)
        img = im.permute(1, 2, 0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
        img = img.copy()
        for box in gt:
            xmin, ymin, xmax, ymax, label = box

            xmin *= img_size
            ymin *= img_size
            xmax *= img_size
            ymax *= img_size
            img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            cv2.putText(img,str(VOC_CLASSES[int(label)]),(int(xmin), int(ymin)-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255))
        cv2.imshow('gt', img)
        cv2.waitKey(0)


