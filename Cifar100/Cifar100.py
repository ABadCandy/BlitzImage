#coding=utf-8
"""
数据集包含100小类，每小类包含600个图像，其中有500个训练图像和100个测试图像。100类被分组为20个大类。每个图像带有1个小类的“fine”标签和1个大类“coarse”标签。
"""
import numpy as np
import os
import cPickle as pickle
import glob  # glob.glob(dir)查找文件只用到三个匹配符："*", "?", "[]"。"*"匹配0个或多个字符；"?"匹配单个字符；"[]"匹配指定范围内的字符，如：[0-9]匹配数字。
import re
import matplotlib.pyplot as plt
import h5py
import sklearn

class DataRead(object):
    data_dir = r'E:\BlitzImage\Cifar100'
    data_dir_cifar100 = os.path.join(data_dir, "cifar-100-python")
    class_names_cifar100 = np.load(os.path.join(data_dir_cifar100, "meta"))

    def one_hot(self, x, n):
        """
        convert index representation to one-hot representation
        """
        x = np.array(x)  # 传入的x为list需转为array
        assert x.ndim == 1
        return np.eye(n)[x]


    def _grayscale(self, a):
        return a.reshape(a.shape[0], 3, 32, 32).mean(1).reshape(a.shape[0], -1)


    def _load_batch_cifar100(self, filename, dtype='float64'):
        """
        load a batch in the CIFAR-100 format
        """
        path = os.path.join(self.data_dir_cifar100, filename)
        batch = np.load(path)
        data = batch['data'] / 255.0
        labels = self.one_hot(batch['fine_labels'], n=100)  # "fine"小类标签,"coarse"大类标签
        return data.astype(dtype), labels.astype(dtype)


    def cifar100(self, dtype='float64', grayscale=False):
        x_train, t_train = self._load_batch_cifar100("train", dtype=dtype)
        x_test, t_test = self._load_batch_cifar100("test", dtype=dtype)

        if grayscale:
            x_train = self._grayscale(x_train)
            x_test = self._grayscale(x_test)

        # save the hdf5 files
        repo_path = os.path.expandvars(os.path.expanduser(self.data_dir))
        save_dir = os.path.join(repo_path, 'HDF5')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fname = os.path.join(save_dir, 'train_data.h5')
        file_train_data = h5py.File(fname, 'w')

        fname = os.path.join(save_dir, 'train_label.h5')
        file_train_label = h5py.File(fname, 'w')

        fname = os.path.join(save_dir, 'test_data.h5')
        file_test_data = h5py.File(fname, 'w')

        fname = os.path.join(save_dir, 'test_label.h5')
        file_test_label = h5py.File(fname, 'w')

        file_train_data.create_dataset('data', data=x_train)
        file_train_data.create_dataset('sample_num', data=x_train.shape[0])
        file_train_label.create_dataset('data', data=t_train)
        file_train_label.create_dataset('sample_num', data=t_train.shape[0])
        file_test_data.create_dataset('data', data=x_test)
        file_test_data.create_dataset('sample_num', data=x_test.shape[0])
        file_test_label.create_dataset('data', data=t_test)
        file_test_label.create_dataset('sample_num', data=t_test.shape[0])

        file_train_data.close()
        file_train_label.close()
        file_test_data.close()
        file_test_label.close()
        return x_train, t_train, x_test, t_test

    def _draw(self):
        Xtrain, Ytrain, Xtest, Ytest = self.cifar100()

        image = Xtrain[0].reshape((3, 32, 32)).transpose(1, 2, 0)
        '''
        image[:,:,0]=Xtrain[0][:1024].reshape(32,32)
        image[:,:,1]=Xtrain[0][1024:2048].reshape(32,32)
        image[:,:,2]=Xtrain[0][2048:].reshape(32,32)
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.axis('off')
        plt.title(self.class_names_cifar100['fine_label_names'][list(Ytrain[0]).index(1)])
        plt.imshow(image)
        plt.show()

    ################################################

    """
    Xtrain = _grayscale(Xtrain)

    image = Xtrain[0].reshape(32, 32)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')
    plt.title(class_names_cifar100['fine_label_names'][list(Ytrain[0]).index(1)])
    plt.imshow(image,cmap='gray')
    plt.show()

    """

    # 图像样本显示
    #  image=np.empty((32,32,3))
    """
    for i in range(32):
        image[:,:,0][i] = Xtrain[0][i*32:i*32+32]

    for j in range(32):
        image[:,:,1][j]=Xtrain[0][1024+j*32:1056+j*32]

    for k in range(32):
        image[:,:,2][k]=Xtrain[0][2048+k*32:2080+k*32]

    print Xtrain[0][:32]
        print image[:,:,0][0]
"""


if __name__ == '__main__':
    dm = DataRead()
    dm.cifar100()
    dm._draw()



