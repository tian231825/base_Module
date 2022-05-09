# -*- encoding: utf-8 -*-
"""
@File    : BasicModule.py
@Time    : 2022/4/24 13:23
@Author  : junruitian
@Software: PyCharm
"""
import time

import torch as t


class BasicModule(t.nn.Module):
    '''
    封装了nn.Module

    '''

    def __init__(self, opt=None):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        '''
        可以指定加载模型的路径

        :param path:
        :return:
        '''
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        '''
        保存模型

        :param name:
        :return:
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M.pth')
        else:
            prefix = 'checkpoints/' + self.model_name + '_' + name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M.pth')
        t.save(self.state_dict(), name)
        return name


class Flat(t.nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        # self.size = size

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # # t = x.size(0)  0->50,1->32,2->7,3->7
        # x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # output = self.out(x)
        # return output, x  # return x for visualization

        # 输出x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，
        # x.size(0) 指batchsize的值
        # view()函数的功能根reshape类似，用来转换size大小。
        # x = x.view(batchsize, -1)中batchsize指转换后有几行，而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
        return x.view(x.size(0), -1)
        # x.view(x.size(0), -1)这句话是说将第二次卷积的输出拉伸为一行，