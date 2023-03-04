
import torch
from IPython import embed
def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []

    while order.numel() > 0:
        try:
            i = order[0]
        except IndexError:
            i = order.item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

def encode_target(target,is_keep_nms):
    """
    :param target: [B,7,7,30] :[B,7,7,2*(dx,dy,w,h,conf,20类]
    :return: boxes:[[x1,y1,x2,y2],.....]
            probs:[]  单个框的置信度
            cls_indexs:[]  ,类别索引
    """

    conf_thresh=0.6 #预测类别的置信度阈值
    grid_num = target.shape[1] #分割格子数量

    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1. / grid_num
    pred = target.data
    pred = pred.squeeze(0)  # 7x7x30
    conf1 = pred[:, :, 4].unsqueeze(2) #第一个框的置信度
    conf2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.cat((conf1, conf2), 2) #[7,7,2]
    mask1 = contain > conf_thresh  #第一种方法：自己设定阈值，如果大于阈值，这个框作为预测框
    mask2 = (contain == contain.max()) #第二种方法，查找置信度最大值，输出
    mask = (mask1 + mask2).gt(0)
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == 1:
                    box = pred[i, j, b * 5:b * 5 + 4]  #预测框位置，[cx,cy,w,h]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])  #预测概率值
                    xy = torch.FloatTensor([j, i]) * cell_size  # cell左上角  up left of cell
                    box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())  # 转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]

                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    boxes.append(box_xy.numpy().tolist())
                    cls_indexs.append(cls_index.item())
                    probs.append(contain_prob.item())

    boxes =torch.tensor(boxes)
    probs=torch.tensor(probs)
    cls_indexs=torch.tensor(cls_indexs)

    keep=nms(boxes,probs)
    if is_keep_nms:
        return boxes[keep],probs[keep],cls_indexs[keep]
    else:
        return boxes,probs,cls_indexs

# import pandas as pd

def out_preds(x):
    x=x.squeeze(0).view(-1,30).numpy().tolist()
    data=pd.DataFrame(x)
    data.to_csv('file.csv')
