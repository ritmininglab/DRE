import os

import torch
from torchvision import datasets, transforms

import torch.multiprocessing
from data.imagenet import ImageNet
import numpy as np 
import pathlib
from PIL import Image
import tqdm
import torchvision.transforms as T
to_pil = T.ToPILImage()
class CustomDataTinyImageNet:
    def __init__(self, dataset, batch_size, transform = None, train = True):
          super(CustomDataTinyImageNet, self).__init__()
          self.dataset = dataset 
          self.transform = transform
          self.train = train 
          self.include_class = list(range(50))
          self.batch_size = batch_size
          self.feat, self.target = None, None 
          self.filter_data()
          

          
          

    def filter_data(self):

            if self.train:
                
                filename = pathlib.Path("Dataset/tinyimagenet/train_data.pt")
            else:
                filename = pathlib.Path("Dataset/tinyimagenet/test_data.pt")

            if not filename.exists():
                if not filename.parent.exists():
                    os.makedirs(filename.parent)
                if self.train:
                    print("Filtering training dataset")
                else:
                    print("Filtering testing dataset")
                loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
                self.feat = []
                self.target = []
                for i, (images, target) in tqdm.tqdm(enumerate(loader), ascii=True, total=len(loader)):
                    
                    for cl in self.include_class:
                        self.feat.extend(images[target==cl])
                        self.target.extend(target[target==cl])
           
                torch.save([self.feat, self.target], filename)
                
            else:
                self.feat, self.target = torch.load(filename)
                print("Datasize", len(self.feat))

    def __getitem__(self, index):
            X = self.feat[index]
            y = self.target[index]
            if self.transform:
                X = to_pil(X)
                X = self.transform(X)
       
                
 
            return X, y

    def __len__(self):
            return len(self.feat)




class TinyImageNet:
    def __init__(self, args):
        super(TinyImageNet, self).__init__()

        data_root = os.path.join(args.data, "tiny_imagenet")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val/images")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )


        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([transforms.ToTensor()])
        )
        train_dataset = CustomDataTinyImageNet(train_dataset, args.batch_size, transform = transform, train = True)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

        transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )



        val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([transforms.ToTensor()])
        )
        val_dataset = CustomDataTinyImageNet(val_dataset, args.batch_size, transform = transform, train = False)

        self.val_loader = torch.utils.data.DataLoader(val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
