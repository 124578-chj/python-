### 1、图像融合
### 2、图像显著性绘制
### 3、图像翻转与镜像
### 4、图像条形码区域分割
### 5、图像处理
### 6、路面质量检测
### 7、表格的识别与修复
### 8、深蹲计算检测
### 9、图像超分辨处理
### 10、调用IP摄像头
### 11、路面坑洼检测
### 12、全景图片的合成
### 13、颜色分割
### 14、图像覆盖
### 15、图像增强
### 16、自动去除背景色
### 17、运动检测-IP摄像头
### 18、阴影图像去除
### 19、车道变道检测
### 20、自然场景下数字检测与识别
### 21、焊接缺陷检测
### 22、人脸动态追踪
### 23、视频中行人消除
### 24、驾驶人员睡意检测
### 25、Swin_yolov1目标检测
这一部分我们尝试让swin_transformer结合最典型的单阶段目标检测算法(Yolov1),用来实现一个目标检测算法。
![Siwn_yolov1框架图](https://github.com/124578-chj/python-image/blob/main/%E9%A1%B9%E7%9B%AE%E7%B4%A0%E6%9D%90/swin_yolov1.png)  
如何开始重新训练这个网络。
运行环境:  
pytorch 1.8.1+cu111  
torchvision  0.9.1+cu111  
opencv-python 45.2.54  
numpy 1.21.5  
visdom  0.1.8.9  

训练/测试数据集：VOC2007,  下载链接：(http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)
把数据集存在一个不包含中文路径的任意文件中。

开始训练，执行train.py  
num_class=20  数据集的类别数  
img_size = 224  图片的长宽尺寸  
batch_size=128  单次训练的数据量  
epochs=2001  所有数据训练的次数  
learning_rate=0.01  初始学习率  
model_weight = r''  预训练模型文件路径，刚开始可不填写，在后续追加训练中，可以填入保存的模型路径    
data_path = r'C:\Users\Administrator\Desktop\VOCdevkit\VOC2007'  数据集的路径   
**运行之前先从终端打开观察端口：python -m visdom.server -p 6006**  





### 26、vit_分类方法应用

