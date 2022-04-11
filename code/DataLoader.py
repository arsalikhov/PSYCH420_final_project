from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm 
from PIL import Image
import torch
import torchvision
import os
import cv2

pylab.rcParams['figure.figsize'] = (8.0, 10.0)
# wrkdir = '/home/arsalikhov/Documents/PSYCH420_final_project/'



class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, coco, ids, cat, labels, classifier = False, transforms=None):
        self.root = root
        self.classifier = classifier
        self.transforms = transforms
        self.coco = coco
        self.ids = ids
        self.labels = labels
        self.cat = cat


    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=self.cat, iscrowd=None)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        img = img.convert('RGB')
        # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # unmasked_img = np.array(unmasked_img)
        # #create_mask
        # mask = coco.annToMask(coco_annotation[0])
        # if len(unmasked_img.shape) == 2:
        #     img = unmasked_img*mask
        # else:
        #     mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
        #     img = unmasked_img*mask

        # img = Image.fromarray(img)
        # number of objects in the image
        num_objs = len(coco_annotation)
        if num_objs > 1:
            pass

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(np.full(num_objs, self.labels), 
        dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype = torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype = torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.classifier:
            my_annotation == my_annotation['labels'][0]


        if self.transforms is not None:
            img = self.transforms(img)

    

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

class MyOwnDataloader:

    def __init__(self, dataDir, dataType, annFile, classes, train_batch_size, classifier = False):

        self.coco = COCO(annFile)
        self.dataDir = dataDir
        self.dataType = dataType
        self.annFile = annFile
        self.classes = classes
        self.train_batch_size = train_batch_size
        self.classifier = classifier

    def get_transform(self):
        custom_transforms = []
        custom_transforms.append(torchvision.transforms.Resize((256, 256)))
        custom_transforms.append(torchvision.transforms.ToTensor())
        custom_transforms.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        return torchvision.transforms.Compose(custom_transforms)

    # collate_fn needs for batch
    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def concat_datasets(self):
        # coco = self.coco
        sub_class_sets = []
        for key, value in self.classes.items():
            ids = self.coco.getCatIds(catNms=[key])
            imgIds = self.coco.getImgIds(catIds=ids)
            if value == 0:
                print(key, value, ids, imgIds[0])
            interim = myOwnDataset(root= self.dataDir + 'images/'+ self.dataType,
                                coco=self.coco,
                                ids = imgIds,
                                cat = ids,
                                labels = value,
                                classifier = self.classifier,
                                transforms=self.get_transform())

            sub_class_sets.append(interim)
        data_sets = torch.utils.data.ConcatDataset(sub_class_sets)
        data_loader =  torch.utils.data.DataLoader(data_sets,
                                                batch_size=self.train_batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                collate_fn = self.collate_fn)

        return data_loader

