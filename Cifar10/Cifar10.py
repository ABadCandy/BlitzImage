#coding=utf-8
import os
import cPickle
import h5py
import numpy as np
import matplotlib.pyplot as plt

def _valid_path_append(path, *args):
    """
    Helper to validate passed path directory and append any subsequent
    filename arguments.

    Arguments:
        path (str): Initial filesystem path.  Should expand to a valid
                    directory.
        *args (list, optional): Any filename or path suffices to append to path
                                for returning.

    Returns:
        (list, str): path prepended list of files from args, or path alone if
                     no args specified.

    Raises:
        ValueError: if path is not a valid directory on this filesystem.
    """
    full_path = os.path.expanduser(path)
    res = []
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    if not os.path.isdir(full_path):
        raise ValueError("path: {0} is not a valid directory".format(path))
    for suffix_path in args:
        res.append(os.path.join(full_path, suffix_path))
    if len(res) == 0:
        return path
    elif len(res) == 1:
        return res[0]
    else:
        return res


def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=False,
                              sqrt_bias=0., min_divisor=1e-8):
    """
    Global contrast normalizes by (optionally) subtracting the mean
    across features and then normalizes by either the vector norm
    or the standard deviation (across features, for each example).

    Parameters
    ----------
    X : ndarray, 2-dimensional
        Design matrix with examples indexed on the first axis and \
        features indexed on the second.

    scale : float, optional
        Multiply features by this const.

    subtract_mean : bool, optional
        Remove the mean across features/pixels before normalizing. \
        Defaults to `True`.

    use_std : bool, optional
        Normalize by the per-example standard deviation across features \
        instead of the vector norm. Defaults to `False`.

    sqrt_bias : float, optional
        Fudge factor added inside the square root. Defaults to 0.

    min_divisor : float, optional
        If the divisor for an example is less than this value, \
        do not apply it. Defaults to `1e-8`.

    Returns
    -------
    Xp : ndarray, 2-dimensional
        The contrast-normalized features.

    """
    assert X.ndim == 2, "X.ndim must be 2"
    scale = float(scale)
    assert scale >= min_divisor

    # Note: this is per-example mean across pixels, not the
    # per-pixel mean across examples. So it is perfectly fine
    # to subtract this without worrying about whether the current
    # object is the train, valid, or test set.
    mean = X.mean(axis=1)  #  无论是axis=0或1得到的mean都为（n,)维度
    if subtract_mean:
        X = X - mean[:, np.newaxis]  # Makes a copy, for example: X:(3,3),mean:(3,), 加了np.newaxis相当于把mean转置了

    else:
        X = X.copy()

    if use_std:
        # ddof=1 simulates MATLAB's var() behaviour, which is what Adam
        # Coates' code does.
        ddof = 1

        # If we don't do this, X.var will return nan.
        if X.shape[1] == 1:
            ddof = 0

        normalizers = np.sqrt(sqrt_bias + X.var(axis=1, ddof=ddof)) / scale
    else:
        normalizers = np.sqrt(sqrt_bias + (X ** 2).sum(axis=1)) / scale

    # Don't normalize by anything too small.
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, np.newaxis]  # Does not make a copy.
    return X

def zca_whiten(Xtrain, Xtest, cachedir):
    num_data1, dim1 = Xtrain.shape
    mean_X1 = Xtrain.mean(axis=0)
    X1 = Xtrain - mean_X1
    sigma1 = (1./num_data1)*(np.dot(X1.T, X1))  # 协方差
    epsilon = 0.1
    U1, S1, V1 = np.linalg.svd(sigma1)
    ZCAWhiten1 = np.dot(np.dot(U1,np.diag(1./np.sqrt(np.diag(S1)+epsilon)) ), U1.T)
    X1 = np.dot(ZCAWhiten1,X1)

    num_data2, dim2 = Xtest.shape
    mean_X2 = Xtest.mean(axis=0)
    X2 = Xtest - mean_X2
    sigma2 = (1. / num_data2) * (np.dot(X2.T, X2))  # 协方差
    epsilon = 0.1
    U2, S2, V2 = np.linalg.svd(sigma2)
    ZCAWhiten2 = np.dot(np.dot(U2, np.diag(1. / np.sqrt(np.diag(S2) + epsilon))), U2.T)
    X2 = np.dot(ZCAWhiten2, X2)

    write_file = open(cachedir, 'wb')
    cPickle.dump(X1, write_file, -1)
    cPickle.dump(X2, write_file, -1)
    write_file.close()
    return X1, X2


def load_cifar10(path=r'E:\ImageProcess\Cifar10', normalize=True, contrast_normalize=False, whiten=False):
    """
    Fetch the CIFAR-10 dataset and load it into memory.

    Args:
        path (str, optional): Local directory in which to cache the raw
                              dataset.  Defaults to current directory.
        normalize (bool, optional): Whether to scale values between 0 and 1.
                                    Defaults to True.

    Returns:
        tuple: Both training and test sets are returned.
    """
    cifar = dataset_meta['cifar-10']
    workdir, filepath = _valid_path_append(path, '', cifar['file'])
    batchdir = os.path.join(workdir, '')

    train_batches = [os.path.join(batchdir, 'data_batch_' + str(i)) for i in range(1, 6)]
    Xlist, ylist = [], []
    for batch in train_batches:
        with open(batch, 'rb') as f:
            d = cPickle.load(f)  # return a dict
            Xlist.append(d['data'])    # data为1个10000*3072大小的uint8s数组。数组的每行存储1张32*32的图像。第1个1024包含红色通道值，下1个包含绿色，最后的1024包含蓝色。图像存储以行顺序为主，所以数组的前32*32列为图像第1行的红色通道值。
            ylist.append(d['labels'])  # 1个10000数的范围为0~9的列表。索引i的数值表示数组data中第i个图像的标签

    X_train = np.vstack(Xlist)  # vstack将Xlist里的6个二维数组按列堆叠成一个二维数组（50000*3072）
    y_train = np.vstack(ylist)

    with open(os.path.join(batchdir, 'test_batch'), 'rb') as f:
        d = cPickle.load(f)
        X_test, y_test = d['data'], d['labels']

    y_train = y_train.reshape(-1, 1) # transfer (5,10000) to(50000,1)
    y_test = np.array(y_test).reshape(-1, 1)
    num_train = y_train.shape[0]
    num_test = y_test.shape[0]

    y_train_new = np.zeros((num_train, 10))
    y_test_new = np.zeros((num_test, 10))
    for col in range(10):
        y_train_new[:, col] = y_train[:,0] == col  # 后面应该是 = =判断
        y_test_new[:, col] = y_test[:,0] == col

    if contrast_normalize:
        norm_scale = 55.0  # Goodfellow
        X_train = global_contrast_normalize(X_train, scale=norm_scale)
        X_test = global_contrast_normalize(X_test, scale=norm_scale)

    if normalize:
        X_train = X_train / 255.
        X_test = X_test / 255.

    if whiten:
        zca_cache = os.path.join(workdir, 'cifar-10-zca-cache.pkl')
        X_train, X_test = zca_whiten(X_train, X_test, zca_cache)


    #save the hdf5 files
    repo_path = os.path.expandvars(os.path.expanduser(workdir))
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

    file_train_data.create_dataset('data', data = X_train)
    file_train_data.create_dataset('sample_num', data = num_train)
    file_train_label.create_dataset('data', data = y_train_new)
    file_train_label.create_dataset('sample_num', data = num_train)
    file_test_data.create_dataset('data', data = X_test)
    file_test_data.create_dataset('sample_num', data = num_test)
    file_test_label.create_dataset('data', data = y_test_new)
    file_test_label.create_dataset('sample_num', data = num_test)

    file_train_data.close()
    file_train_label.close()
    file_test_data.close()
    file_test_label.close()

    return (X_train, y_train_new), (X_test, y_test_new), 10


dataset_meta = {
    'cifar-10': {
        'size': 170498071,
        'file': 'cifar-10-python.tar.gz',
        'url': 'http://www.cs.toronto.edu/~kriz',
        'func': load_cifar10
    }
}

if __name__ == '__main__':
    (Xtrain,Xlabel),(Ytest,Ylabel),clas=load_cifar10()
    print Xtrain.shape,Xlabel.shape,Ytest.shape,Ylabel.shape
    image = Xtrain[0].reshape((3, 32, 32)).transpose(1, 2, 0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis('off')
    plt.imshow(image)
    plt.show()
