from model import ViT,DeepViT
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from torch.autograd import Variable
import torch.optim as optim
import torchvision
from visdom import Visdom
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision import transforms,models
from utils import GradCAM,  center_crop_img
import cv2
def show_cam_on_image(img: torch.tensor,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.[B,3,224,224]
    :param mask: [B,1,224,224]
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    B,_,W,H=img.shape
    img=img.detach().cpu().permute(0,3, 2, 1).numpy()

    final_out=[]
    for i in range(B):
        im=img[i]
        mas=mask[i]
        heatmap = cv2.applyColorMap(np.uint8(255 * mas), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        if np.max(img) > 1:
            raise Exception(
                "The input image should np.float32 in the range [0, 1]")
        cam = heatmap + im
        cam = cam / np.max(cam)
        final_out.append(cam)
    final_out=torch.tensor(np.array(final_out))
    return final_out.permute(0,3,2,1)
class ReshapeTransform:
    def __init__(self, model):
        input_size = (256,256)
        patch_size = (32,32)
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


normalize = transforms.Normalize(mean=[0.485], std=[0.229])
transform_train = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.Pad(10),
    transforms.RandomCrop((256, 256)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


class Net(nn.Module):
    def __init__(self,num_class):
        super(Net, self).__init__()
        self.model=DeepViT(img_size = 256,
        patch_size = 32,
        dim = 1024,
        depth = 3,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
            channels=1)
        self.fc=nn.Linear(1024,num_class)

    def forward(self,x):
        x=self.model(x)
        x=self.fc(x)
        return x

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

def train_one_epoch(epoch,model,criterion,use_gpu=True,visdom=None):
    model.train()
    if use_gpu:
        model.to('cuda')
        criterion.to('cuda')
    for i, (img, label) in enumerate(trainloader):
        if use_gpu:
            img = Variable(img.cuda())
            label = Variable(label.cuda())
        label= torch.as_tensor(label, dtype=torch.long)
        pred = model.forward(img)

        target_category =label.T.detach().cpu().tolist()

        grayscale_cam = cam(input_tensor=img, target_category=target_category)

        rgb_cam = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        loss = criterion(pred, label)
        loss_all.update(loss.item())

        optimizer_P.zero_grad()
        loss.backward()
        optimizer_P.step()

        rgb_cam = rgb_cam.detach().cpu().numpy()
        viz.images(rgb_cam, win='gen_img')

        if i%50==0:
            print('Epoch[{}],[{}/{}]  loss: {:.3f}'.format(epoch,i,len(trainloader),loss.item()))
    visdom.line(Y=[loss_all.avg], X=[epoch], win='train_loss', update='append', opts={
        'showlegend': True,  # 显示网格
        'title': "Trian loss",
        'xlabel': "time",  # x轴标签
        'ylabel': "loss",  # y轴标签
        'linecolor': np.array([[0, 125, 255]]),
        'legend': ['loss']
    }, )
    if epoch % 5 == 0 and epoch != 0:
        model.eval()
        for j, (img, label) in enumerate(trainloader):
            if use_gpu:
                img = Variable(img.cuda())
            pred = model(img)
            bcc = accuracy(pred.cpu(), label)
            train_acc.update(bcc[0].item(), label.size(0))
        visdom.line(Y=[train_acc.avg], X=[epoch], win='train_acc', update='append', opts={
            'showlegend': True,  # 显示网格
            'title': "Trian Acc",
            'xlabel': "time",  # x轴标签
            'ylabel': "acc",  # y轴标签
            'linecolor': np.array([[125, 0, 255]]),
            'legend': ['acc']
        }, )
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class vis_attn():
    def __init__(self):
        self.grad_block=[]
        self.fmap_block=[]

    def backward_hook(self,module, grad_in, grad_out):
        self.grad_bloc.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def farward_hook(self,module, input, output):
        self.fmap_block.append(output)



if __name__=='__main__':
    epochs = 101
    learning_rate = 0.0001
    num_class = 10
    batchs=64
    use_gpu=True
    pretrained=True
    weight_path='deep_vit.pt'

    train_acc = AverageMeter()
    val_acc = AverageMeter()
    loss_all=AverageMeter()

    train_data=torchvision.datasets.MNIST(root='',train = True, transform= transform_train,download=True)
    trainloader = data.DataLoader(train_data, batch_size=batchs, shuffle=True, drop_last=True)

    model=Net(num_class)
    if pretrained:
        print('loading Saving Weight....')
        model.load_state_dict(torch.load(weight_path))


    target_layers = [model.model.encoder.layers[-1][-1].norm]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=ReshapeTransform(model))






    optimizer_P = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # 损失

    viz = Visdom(env='demo', port=1001)

    for epoch in range(epochs):
        train_one_epoch(epoch, model, criterion, use_gpu, visdom=viz)
        torch.save(model.state_dict(), 'deep_vit.pt')

