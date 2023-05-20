#!/usr/bin/env python3
import os
from typing import Any
import numpy as np
import torch
from torchvision import datasets
import copy
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from pycocotools.coco import COCO

class CoCoDataset(datasets.VisionDataset):
    """
    This is the class for the COCO dataset. It is a subclass of the VisionDataset class.
    This is Manually written from scratch. 
    source: https://youtu.be/Uc90rr5jbA4
    
    Inputs:
        root:
        annFile:
        transform:
        Target_transform:
        transforms:
    """
    def __init__(self, root:str, annFile:str, transform=None, target_transform=None, transforms=None):
        super(CoCoDataset, self).__init__(root, transforms, transform, target_transform)
        self.root = root            # root directory for images
        self.coco = COCO(annFile)   # loading annotation file
        self.ids = list(sorted(self.coco.imgs.keys()))                          # list of image ids
        self.ids = [id for id in self.ids if len(self._load_targets(id)) > 0]   # Filtering out images with no annotations
        
        
    def _load_images(self, id:int):
        img_path = self.coco.loadImgs(id)[0]['file_name']
        relative_path = os.path.join(self.root, img_path)
        image = np.array(Image.open(relative_path).convert('RGB'))
        # image = np.array(Image.open(relative_path))
        return image
    
    def _load_targets(self, id:int):
        """
        This method loads how many annotations in an image:id
        eg: for image:1, there are 2 annotations (2 fish)
        getAnnIds(1) returns [1,2]
        loadAnns([1,2]) returns 2x [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [x,y,width,height]}...]
        
        Inputs:
            id: image id(int)
        Returns:
            annotaions: list or dict
        """
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        # loading images and targets from dataset
        id = self.ids[index]
        image = self._load_images(id)
        target = copy.deepcopy(self._load_targets(id))
        
        # Transforming images and targets
        boxes = [t['bbox'] + [t['category_id']] for t in target]   # Extracting bounding boxes and labels from annotations
        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
        image, boxes = transformed['image'], transformed['bboxes']
        
            
        # Converting boxes from [x,y,width,height] to [x1,y1,x2,y2]
        new_boxes = []
        for box in boxes:
            new_boxes.append([box[0], box[1], box[0]+box[2], box[1]+box[3]])
        boxes = torch.as_tensor(new_boxes, dtype=torch.int64)
        
        
        # Returning image and targets
        targ = {}
        targ['image_id']= torch.tensor([t['image_id'] for t in target], dtype=torch.int64)
        targ['labels']  = torch.tensor([t['category_id'] for t in target], dtype=torch.int64)
        targ['boxes']   = boxes
        targ['area']    = torch.tensor([t['area'] for t in target], dtype=torch.float32)
        targ['iscrowd'] = torch.tensor([t['iscrowd'] for t in target], dtype=torch.int64)
        return image.div(255), targ
    
    
class CustomFunctions:
    def __init__(self):
        super(CustomFunctions, self).__init__()
        pass
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    def visualise_image(data, lable_names):
        image = torch.tensor(data[0]*255, dtype=torch.uint8)
        boxes = data[1]['boxes']
        classes = [lable_names[i] for i in data[1]['labels']]
        return draw_bounding_boxes(image, boxes, classes, width=2, font_size=20).permute(1,2,0)
    
    
        
        