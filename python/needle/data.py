import gzip
import struct
from typing import Any, Iterable, Iterator, List, Optional, Sized, Union

import numpy as np

from .autograd import Tensor

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filename, 'rb') as img_file:
        magic, nums, rows, cols = struct.unpack('>IIII', img_file.read(16))
        images = np.fromstring(img_file.read(), dtype=np.uint8).reshape(nums,-1).astype(np.float32)
    _range = np.max(images) - np.min(images)
    images = (images - np.min(images)) / _range

    with gzip.open(label_filename, 'rb') as label_file:
        magic, nums = struct.unpack('>II', label_file.read(8))
        labels = np.fromstring(label_file.read(), dtype=np.uint8)

    return images, labels

class Transform:
    def __call__(self, x):
        raise NotImplementedError

class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            # 水平翻转 H x W x C格式的 NDArray
            return np.flip(img, 1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    """
    RandomCrop 不改变输入图片的大小
    padding 定义好裁剪框的移动范围，裁剪偏移随机落在[-padding, padding]当中
    """
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below。
            一个裁剪框和图片重叠，然后把这个裁剪框在H 和 W两个方向平移 shift_x 和 shift_y
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        # 填充0
        img_pad = np.zeros_like(img)
        H, W = img.shape[0], img.shape[1]
        if abs(shift_x) >= H or abs(shift_y) >= W:
            return img_pad
        
        img_start_x, img_end_x = max(0, shift_x), min(H, H + shift_x)
        img_start_y, img_end_y = max(0, shift_y), min(W, W + shift_y)
        pad_start_x, pad_end_x = max(0, -shift_x), min(H, H - shift_x)
        pad_start_y, pad_end_y = max(0, -shift_y), min(W, W - shift_y)
        img_pad[pad_start_x: pad_end_x, pad_start_y: pad_end_y, :] = img[img_start_x: img_end_x, img_start_y: img_end_y, :]
        return img_pad
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        indices = np.arange(len(dataset))

        if not self.shuffle:
            # range(batch_size, len(dataset), batch_size) 能确定好每个batch第一个元素的下标。
            # array_split 以这些下标位置作为截断点
            self.ordering = np.array_split(indices, 
                                           range(batch_size, len(dataset), batch_size))
        else:
            # 打乱顺序，shuffle方法时in-place的
            np.random.shuffle(indices)
            # 只需要打乱
            self.ordering = np.array_split(indices,
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.start = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        """
            ordering长这样：[array([5, 1, 7]), array([6, 2, 0]), array([3, 4])]
        """
        ### BEGIN YOUR SOLUTION
        # 检查iter是不是走完了
        if self.start >= len(self.ordering):
            raise StopIteration
        indices = self.ordering[self.start]
        self.start += 1
        samples = [Tensor(x) for x in self.dataset[indices]]
        return tuple(samples)
        ### END YOUR SOLUTION

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        assert len(self.images) == len(self.labels) , "the number of images should be the same with the label's"
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # image过一遍transformes
        image = self.images[index]
        # label过一遍transformes
        label = self.labels[index]
        
        if self.transforms is not None:
            image = image.reshape((28, 28, -1))
            image = self.apply_transforms(image)
            image_ret = image.reshape((-1, 28, 28))
            return image_ret, label
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.labels.shape[0]
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
