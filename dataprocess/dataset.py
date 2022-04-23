from torchvision import transforms
import os
import time
from torch.utils import data
import numpy as np
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, root, input_size = 224, trans=None, train=True, test=False):
        self.test = test
        self.train = train
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        '''
    the format of test and trian image name is different
    as for test: /test/102.jpg
    as for train: /train/cat.1.jpg
    '''
        if test:  # root: './dogvscat/test/' imgs = ["xx/123.jpg", "xx/234.jpg", ...]
            sorted(imgs, key=lambda x: int(x.split(".")[-2].split("/")[-1]))
        else:
            sorted(imgs, key=lambda x: int(x.split(".")[-2]))

        # shuffle
        np.random.seed(100)
        imgs = np.random.permutation(imgs)

        # split dataset
        self.imgs = imgs

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # test and dev dataset do not need to do data augemetation
        if self.test or not self.train:
            self.trans = transforms.Compose([
                transforms.Resize(int(input_size/0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.trans = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        '''
        as for test: just return the id of picture.
        as for train and dev: return 1 if dog, return 0 if cat
        '''
        imgpath = self.imgs[index]
        if self.test:
            label = int(imgpath.split(".")[-2].split("/")[-1])
        else:
            kind = imgpath.split(".")[-3].split("/")[-1]
            label = 1 if kind == "good" else 0
        img = Image.open(imgpath)
        img = self.trans(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
