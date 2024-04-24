import numpy as np


class RunningMeanStd:
    # 这个类用于动态地计算输入数据的均值和标准差。
    def __init__(self, shape):  # 初始化方法，shape参数定义了输入数据的维度。
        self.n = 0  # 初始化数据点的计数为0。
        self.mean = np.zeros(shape)  # 初始化均值为0，其维度由shape参数指定。
        self.S = np.zeros(shape)  # 初始化用于计算标准差的中间变量S。
        self.std = np.sqrt(self.S)  # 计算标准差，此时因为S为0，标准差也为0。

    def update(self, x):
        x = np.array(x)  # 确保输入数据x为NumPy数组。
        self.n += 1  # 更新数据点计数。
        if self.n == 1:
            self.mean = x  # 如果是第一个数据点，直接设置均值和标准差。
            self.std = np.zeros(x.shape)
        else:
            old_mean = self.mean.copy()  # 复制当前均值，用于计算新的均值和S。
            self.mean = old_mean + (x - old_mean) / self.n  # 更新均值。
            self.S = self.S + (x - old_mean) * (x - self.mean)  # 更新S。
            self.std = np.sqrt(self.S / self.n)  # 计算新的标准差。


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)  # 创建RunningMeanStd实例。

    def __call__(self, x, update=True):
        # 标准化处理函数。update参数控制是否更新均值和标准差。
        if update:
            self.running_ms.update(x)  # 更新均值和标准差。
        x = (x - self.running_ms.mean) / \
            (self.running_ms.std + 1e-8)  # 标准化处理，避免除零错误。
        return x  # 返回标准化后的数据。


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # 奖励的维度，通常为1。
        self.gamma = gamma  # 折扣因子。
        self.running_ms = RunningMeanStd(
            shape=self.shape)  # 创建RunningMeanStd实例。
        self.R = np.zeros(self.shape)  # 初始化累计奖励为0。

    def __call__(self, x):
        self.R = self.gamma * self.R + x  # 更新累计奖励。
        self.running_ms.update(self.R)  # 更新累计奖励的均值和标准差。
        x = x / (self.running_ms.std + 1e-8)  # 奖励缩放，仅使用标准差。
        return x  # 返回缩放后的奖励。

    def reset(self):
        self.R = np.zeros(self.shape)  # 当一个episode结束时，重置累计奖励。
