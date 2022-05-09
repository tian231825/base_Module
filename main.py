# -*- encoding: utf-8 -*-
"""
@File    : main.py
@Time    : 2022/4/22 11:22
@Author  : junruitian
@Software: PyCharm
"""
import csv
import os
import fire
import torch as t
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchnet import meter
import models
from config import opt
from data.dataset import DogCat
from utils.visualize import Visualizer
from tqdm import tqdm

# opt = DefaultConfig()


def train(**kwargs):
    '''

    :param kwargs:
    :return:
    '''
    # 根据命令行参数修改参数
    opt.parse(kwargs)
    # vis = Visualizer(opt.env)

    # step 1: 模型
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step 2: 数据
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # step 3:目标函数和优化器
    '''
    t.nn.CrossEntropyLoss() 
    交叉熵损失函数:用于解决多分类问题，也可用于解决二分类问题
    其内部会自动加上Sofrmax层
    '''
    criterion = t.nn.CrossEntropyLoss()

    # 学习率
    lr = opt.learning_rate

    '''
    Adam优化算法
    Adam优化算法基本上就是将momentum和rmsprop结合在一起
    Adam代表的是adaptive moment estimation,本质上是带有动量项的RMSprop，
    它利用梯度的一阶矩阵估计和二阶矩估计动态调整每个参数的学习率。
    它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
    https://blog.csdn.net/weixin_38145317/article/details/104775536
    优化器需要知道当前的网络或者别的什么模型的参数空间，这也就是为什么在训练文件中，
    ***正式开始训练之前需要将网络的参数放到优化器中***
    *** weight_decay,可选，权重衰减，L2乘法，默认0 ***
    参数：
    params(iterable)--待优化参数的iterable或者是定义了参数组的dict
    lr (float,可选)，学习率(步长因子），默认le-3=0.001，控制了权重的更新比率.较大的值（如织0.3)在学习率更新前会有更快地初始学习，
        而较小的值如le-5会令训练收敛到更好的性能。
    betas=[beta1,beta2],可选，用于计算梯度以及梯度平方的运行平均值的系数，默认为[0.9,0.999],
        beta1是一阶矩阵的指数衰减率，beta2是二阶矩阵的指数衰减率，该超参数在稀疏梯度（如在NLP或计算机视觉任务中）应该设置为接近1的数。
    eps,epsion,该参数是非常小的数，为了增加数值计算的稳定性而加到分母里的项，默认le-8，为了防止在实现中除以零；
    weight_decay,可选，权重衰减，L2乘法，默认0
    '''
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step 4:统计指标：平滑处理后的损失，还有混淆矩阵

    '''
    meter.AverageValueMeter()
    函数作用：添加单值数据，进行取平均值及标准差计算。(标准差是方差的算数平方根)
    通过.value查询
    example: 
    # a = AverageValueMeter()
    # a.add(1)
    # print(a.value())
    # a.add(2)
    # print(a.value())
    # a.add(3)
    # print(a.value())
    result:
    # (1.0, inf)
    # (1.5, 0.7071067811865476)
    # (2.0, 1.0)
    '''
    loss_meter = meter.AverageValueMeter()
    '''
    confusion_matrix = meter.ConfusionMeter(class_num) #指定类别数目
    *** necessary-混淆矩阵 ***
    '''
    confusion_matrix = meter.ConfusionMeter(2)
    # previous_loss  实际上是一个指标，终止条件， 即如果损失不再下降，则降低学习率
    previous_loss = 1e100

    # 训练
    for epoch in range(opt.max_epoch):
        '''
        使用重置(清空序列)：loss_meter.reset()
        为了可视化增加的内容
        每个epoch开始前，将存放的loss清除，重新开始记录
        '''
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in enumerate(train_dataloader):
            # 训练模型

            input = Variable(data)
            target = Variable(label)

            # input = data
            # target = label
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            '''
            optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0
            在学习pytorch的时候注意到，对于每个batch大都执行了这样的操作：
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            对于这些操作我是把它理解成一种梯度下降法
            https://blog.csdn.net/scut_salmon/article/details/82414730
            '''
            # 即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
            optimizer.zero_grad()
            # outputs = net(inputs) 即前向传播求出预测的值
            score = model(input)
            # 这一步很明显，就是求loss（其实我觉得这一步不用也可以，反向传播时用不到loss值，只是为了让我们知道当前的loss是多少）
            loss = criterion(score, target)
            # 即反向传播求梯度
            loss.backward()
            # 即更新所有参数
            optimizer.step()
            '''
            https://blog.csdn.net/weixin_38145317/article/details/104775536
            当前参数空间对应的梯度，optimzier使用之前需要zero清零一下，如果不清零，那么使用的这个grad就得通上一个mini-batch有关，
            这不是我们需要的结果。
            我们知道optimizer更新参数空间需要基于反向梯度，因此，当调用optimizer.step()的时候应当是loss.backward()的时候，
            loss.backward()在前，然后跟一个step.
            optimizer.step()需要放在每一个batch训练中，而不是epoch训练中，
            这是因为现在的mini-batch训练模式是假定每一个训练集就只有mini-batch这样大，
            因此就可以将每一次mini-batch看作是一次训练，一次训练更新一次参数空间，因而optimizer.step()放在这里。
            只有用了optimizer.step()，模型才会更新
            '''

            # 更新统计指标以及可视化
            '''
            使用1.7.1版本的torch框架运行代码时出现问题报错如下：
            invalid index of a 0-dim tensor. Use tensor.item() in Python or tensor.item<T>() in C++ to convet
            源代码是在1.4.0版本的torch下运行的。
            'losses.update(loss.data[0], input.size(0))' '在torch0.4的版本使用此句，高一些的版本用下面的用法'
            ---  losses.update(loss.item(), input.size(0))
            'top1.update(prec1[0], input.size(0))'
            ---  top1.update(prec1.item(), input.size(0))

            '''
            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, target.data)

            if ii % opt.print_freq == opt.print_freq - 1:
                # TODO
                # vis.plot('loss', loss_meter.value()[0])

                # 如果需要的话，进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        # model_save_name_path = "{:03d}-{:.2f}".format(epoch, val_accuracy)
        # model.save(model_save_name_path)

        # 计算验证集上的指标及可视化
        # TODO
        val_cm, val_accuracy = val(model, val_dataloader)
        # vis.plot('val_accuracy', val_accuracy)
        # vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
        #     epoch=epoch,
        #     loss=loss_meter.value()[0],
        #     val_cm=str(val_cm.value()),
        #     train_cm=str(confusion_matrix.value()),
        #     lr=lr))


        # 如果损失不再下降，则降低学习率
        '''
        在训练中动态的调整学习率
        # 新建optimizer或者修改optimizer.params_groups对应的学习率
        # # 新建optimizer更简单也更推荐，optimizer十分轻量级，所以开销很小
        # # 但是新的优化器会初始化动量等状态信息，这对于使用动量的优化器（momentum参数的sgd）可能会造成收敛中的震荡
        # ## optimizer.param_groups:长度2的list，optimizer.param_groups[0]：长度6的字典
        print(optimizer.param_groups[0]['lr'])
        old_lr = 0.1
        optimizer = optim.SGD([{'params': net.features.parameters()},
                {'params': net.classifiter.parameters(), 'lr': old_lr*0.1}], lr=1e-5)
        可以看到optimizer.param_groups结构，[{'params','lr', 'momentum', 'dampening', 'weight_decay', 'nesterov'},{……}]，集合了优化器的各项参数。
        '''
        '''
        动态修改学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr 
        
        得到学习率   optimizer.param_groups[0]["lr"] 
        '''
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.learning_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

        # 为了记录参数 将model.save转移到val之后
        model_save_name_path = "{:03d}-{:.2f}".format(epoch, val_accuracy)
        matrix_value = val_cm.value()
        # TODO 中间结果输出
        print("Epoch [{:03d}/{:03d}], Loss: {:.4f}, correct:{}, total:{}, Training Accuracy: {:.4f} %"
              .format(epoch, opt.max_epoch, loss_meter.value()[0], (matrix_value[0][0]+matrix_value[1][1]),
                      matrix_value.sum(), (matrix_value[0][0]+matrix_value[1][1])/matrix_value.sum()))

        model.save(model_save_name_path)


def val(model, dataloader):
    '''
    计算模型在验证集上的准确率等信息，用以辅助训练
    验证完成后还需要将其置回为训练模式(model.train())，这两句代码会影响BatchNorm和Dropout等层的运行模式。
    '''

    '''
    model.eval()
    model.eval() 作用等同于 self.train(False)
    简而言之，就是评估模式。而非训练模式。
    在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移
    '''

    # 把模型设为验证模式
    model.eval()

    confusion_matrix = meter.ConfusionMeter(2)
    '''
    reset方法用于将混淆矩阵重新赋值为全0
    '''
    for ii, data in enumerate(dataloader):
        input, label = data
        '''
        pytorch进行自动梯度下降时出现错误
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)
        由于版本迭代，我们只需要按照提示步骤将其改为如下表示方式即可：
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
            with torch.no_grad():
                target_var = torch.autograd.Variable(target)
        '''
        with t.no_grad():
            val_input = t.autograd.Variable(input)
        with t.no_grad():
                val_label = t.autograd.Variable(label.long())
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)

        confusion_matrix.add(score.data.squeeze(), label.long())

    # 把模型恢复为训练模式
    model.train()

    '''
    meter提供了一些轻量级的工具，用于帮助用户快速统计训练过程中的一些指标。
    AverageValueMeter能够计算所有数的平均值和标准差，这里用来统计一个epoch中损失的平均值。
    confusionmeter用来统计分类问题中的分类情况，是一个比准确率更详细的统计指标。
    混淆矩阵：
    样本      判为a       判为b
    实际为a    35          15
    实际为b     9          91
    # meter 中confusion_matrix结构是多维方阵，这里面accuarcy计算方式对应样本数
    '''
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    # PRF三个指标
    precision = 100. * (cm_value[0][0]) / (cm_value[0][0] + cm_value[1][0])
    recall = 100. * (cm_value[0][0]) / (cm_value[0][0] + cm_value[0][1])
    F1score = 100. * 2 * precision * recall / (precision + recall)
    return confusion_matrix, accuracy


def write_csv(results, file_name):
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


def test(**kwargs):
    '''
    测试(inference)
    :param kwargs:
    :return:
    '''

    '''
    测试时，需要计算每个样本属于狗的概率，并将结果保存成csv文件。测试的代码与验证比较相似，但需要自己加载模型和数据。
    '''
    opt.parse(kwargs)

    # 模型
    # configure model
    model = getattr(models, opt.model)()
    model = model.eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # 数据
    test_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    results = []

    for ii, (data, path) in enumerate(test_dataloader):
        # TODO
        with t.no_grad():
            input = t.autograd.Variable(data)
        # input = t.autograd.Variable(data, volatile=True)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        # TODO
        probability = t.nn.functional.softmax(score, dim=1)[:1].data.tolist()
        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]
        results += batch_results

    write_csv(results, opt.result_file)

    return results


if __name__ == '__main__':
    # model = getattr(models, 'AlexNet')

    # learning_rate = opt.learning_rate
    # model = getattr(models, opt.model)
    # dataset = DogCat(opt.train_data_root)
    # print(1)
    fire.Fire()