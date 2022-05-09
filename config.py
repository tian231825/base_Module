# -*- encoding: utf-8 -*-
"""
@File    : config.py
@Time    : 2022/4/22 11:22
@Author  : junruitian
@Software: PyCharm
"""
import warnings


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model = 'ResNet34'

    train_data_root = 'data/train/'
    test_data_root = 'data/test/'
    load_model_path = None

    batch_size = 128
    use_gpu = True
    num_workers = 4
    print_freq = 20

    debug_file = 'tmp/debug'
    result_file = 'result.csv'

    max_epoch = 10
    learning_rate = 0.1
    # when val_loss increase, lr=lr*decay
    learning_decay = 0.95
    weight_decay = 1e-4


    '''
    parse 应用:
    opt = DefaultConfig()
    new_config = {'learning_rate' = 0.1, 'use_gpu' = False}
    opt.parse(new_config)    
    '''


def parse(self, kwargs):
    '''
    根据字典更新config参数
    :param kwargs:
    :return:
    '''
    # 更新配置参数
    for k, v in kwargs.items():
        if not hasattr(self, k):
            # 这里报错或者警告
            warnings.warn("Warning: opt has not attribute %s" % k)
        setattr(self, k, v)
    # 打印配置信息
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))


DefaultConfig.parse = parse
opt = DefaultConfig()
# opt.parse = parse
