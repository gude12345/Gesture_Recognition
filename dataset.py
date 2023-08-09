from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import os


class MyData(data.Dataset):
    def __init__(self, transforms=None, train=True, data_dir=None):
        self.images_path = []
        self.labels = []
        self.read_file(data_dir)
        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.Resize((140, 140)),
                    T.CenterCrop(128),
                    T.ToTensor()
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ColorJitter(brightness=0.2),
                    T.Resize((140, 140)),
                    T.RandomCrop(128),
                    T.ToTensor()
                ])

    def read_file(self, data_dir):
        data_name = ['backward', 'backwardForward', 'forward', 'forwardBackward', 'forwardLeft',\
                     'forwardRight', 'left', 'leftRight', 'right', 'rightLeft']
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.rsplit('.')[-1] == 'jpg':
                    label = int(data_name.index(root.rsplit('//')[-1]))
                    img_path = os.path.join(root, file)
                    self.images_path.append(img_path)
                    self.labels.append(label)

    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = self.labels[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        data = self.transforms(img)
        return data, int(label)

    def __len__(self):
        return self.images_path.__len__()
