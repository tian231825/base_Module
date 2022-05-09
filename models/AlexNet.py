# -*- encoding: utf-8 -*-
"""
@File    : AlexNet.py.py
@Time    : 2022/4/25 14:10
@Author  : junruitian
@Software: PyCharm
"""
from torch import nn
from .BasicModule import BasicModule


class AlexNet(BasicModule):

    '''
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    '''

    # 解释
    # https://blog.csdn.net/pengchengliu/article/details/108909195

    # 此处 num_classes 为分类的数量
    def __init__(self,  num_classes=2):
        '''

        封装函数
        '''
        super(AlexNet, self).__init__()
        self.model_name = 'Alexnet'

        self.features = nn.Sequential(
            # 二维卷积 nn.Conv2d
            # nn.Conv2d:常用在图像 (batch,channel,height,width)（批数量，通道数，高度，长度）
            # in_channels 是输入的四维张量[N, C, H, W]中的C了，即输入张量的channels数。这个形参是确定权重等可学习参数的shape所必需的。
            # 对于含有RGB 3个通道的彩色图片，每张图片包含了h行w列像素点，每个点需要3个数值表示RGB通道的颜色强度，因此一张图片可以表示为[h, w, 3]
            # kernel_size 卷积核的大小，一般我们会使用5x5、3x3这种左右两个数相同的卷积核，
                # 因此这种情况只需要写kernel_size = 5这样的就行了。
                # 如果左右两个数不同，比如3x5的卷积核，那么写作kernel_size = (3, 5)，注意需要写一个tuple，而不能写一个列表（list）
            # stride=1 卷积核在图像窗口上每次平移的间隔，即所谓的步长。
            # padding = (kernel_size / 2)down 向下取整 如果kernel——size是个tuple 那么padding也是
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # RELU 修正线性单元 RELU(x)=max(0,x)
            # 神经网络 P84 Chapter4.1.2
            # [Nair et al., 2010], [He et al., 2015], [Maas et al., 2013], [Clevert et al., 2015], [Dugas et al., 2001]
            # inplace = False 时,不会修改输入对象的值,而是返回一个新创建的对象,所以打印出对象存储地址不同,类似于C语言的值传递
            # inplace = True 时,会修改输入对象的值,所以打印出对象存储地址相同,类似于C语言的址传递,
                # 会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
            nn.ReLU(inplace=True),
            # 卷积操作中 pool层是比较重要的，是提取重要信息的操作，可以去掉不重要的信息，减少计算开销
            # kernel_size(int or tuple) - max pooling的窗口大小  ，
            # stride(int or tuple, optional) - max pooling的窗口移动的步长。默认值是kernel_size
            # padding(int or tuple, optional) - 输入的每一条边补充0的层数
            # dilation(int or tuple, optional) – 一个控制窗口中元素步幅的参数 例如3*3中2*2的的kernel_size中,dilation为1，则是四个角
            # return_indices - 如果等于True，会返回输出最大值的序号，对于上采样操作会有帮助
            # ceil_mode - 如果等于True，计算输出信号大小的时候，会使用向上取整，代替默认的向下取整的操作
            nn.MaxPool2d(kernel_size=3, stride=2),
            # pool操作并不改变张量的通道数
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )
        self.classifer = nn.Sequential(
            # nn.dropout()是为了防止或减轻过拟合而使用的函数，它一般用在全连接层
            # Dropout就是在不同的训练过程中随机扔掉一部分神经元。也就是让某个神经元的激活值以一定的概率p，让其停止工作，
            # 这次训练过程中不更新权值，也不参加神经网络的计算。但是它的权重得保留下来（只是暂时不更新而已），
            # 因为下次样本输入时它可能又得工作了
            nn.Dropout(),
            # 全连接层。输入的节点数是256 * 6 * 6。因为前面输入的是特征图，特征图是6 * 6大小的，有256个特征图。
            # 自己有4096个神经元。参数个数：(256 * 6 * 6 + 1) * 4096 = 37, 752, 832
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # 全连接层，输入节点数是4096，输出节点数4096。参数个数：4097*4096 = 16,781,312
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x.view()就是对tensor进行reshape
        # 将向量铺平，便于传入全连接层
        x = x.view(x.size(0), 256*6*6)
        x = self.classifer(x)
        return x
