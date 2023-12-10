import sys
sys.path.append("./python")
import needle as ndl
import numpy as np

def main():

    mnist_train_dataset = ndl.data.MNISTDataset("data/train-images-idx3-ubyte.gz",
                                                "data/train-labels-idx1-ubyte.gz")
    data = mnist_train_dataset[np.arange(3)]
    print(data)
    
if __name__ == '__main__':
    main()