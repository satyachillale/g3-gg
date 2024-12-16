import transformers
import torch
import os
import numpy as np
import tarfile
import pickle
from tqdm import tqdm
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPModel
from torchvision.datasets import VisionDataset
from typing import Callable, Optional
from torchvision.io import ImageReadMode, read_image
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
from io import BytesIO
from PIL import ImageFile
from torch.utils.data import get_worker_info
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow truncated images to be loaded
import webdataset as wds

class MP16Dataset(VisionDataset):

    def __init__(self, root_path='./data/', text_data_path='remaining_dataset.csv', image_data_path='mp-16-images.tar', vision_processor=None,text_processor=None):
        super().__init__(self)

        # Initialize paths and metadata
        self.root_path = root_path
        self.text_data_path = os.path.join(root_path, text_data_path)
        self.image_data_path = os.path.join(root_path, image_data_path)
        self.text_data = pd.read_csv(self.text_data_path)

        # Preprocess text data
        self.text_data['IMG_ID'] = self.text_data['IMG_ID'].apply(lambda x: x.replace('/', '_'))
        self.text_data = self.text_data[self.text_data['country'].notnull()]
        print('Text data loaded:', self.text_data.shape[0])

        # Convert longitude and latitude
        self.text_data['LON'] = self.text_data['LON'].astype(float)
        self.text_data['LAT'] = self.text_data['LAT'].astype(float)

        # Image transformations
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0
        print(f"World Size: {world_size}, Rank: {rank}")

        # Initialize WebDataset for tar file
        self.wds_pipeline = (
            wds.WebDataset(self.image_data_path)
            .shard(rank, world_size)
            .decode("pil")  # Automatically decode images to PIL
            .to_tuple("jpg", "__key__")  # Extract image and key (filename)
        )
        
        # Optional: Add vision_processor
        self.vision_processor = vision_processor
        self.text_processor = text_processor
        
        # Define contrast transforms (for augmentations)
        self.contrast_transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(size=224),
            T.RandomApply([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=9),
            T.ToTensor()
        ])

    def caption_generation(self, row):
        pass

    def __getitem__(self, index):
        # Fetch metadata for the current sample
        row = self.text_data.iloc[index]
        img_id = row['IMG_ID']
        longitude = row['LON']
        latitude = row['LAT']
        
        # Generate textual description
        location_elements = [row[col] for col in ['neighbourhood', 'city', 'state', 'country'] 
                             if col in row and pd.notna(row[col])]
        text = 'A street view photo taken in ' + ', '.join(location_elements)
        
        # Load image from tar using WebDataset
        for image, key in self.wds_pipeline:
            if key == img_id:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                break
        
        # Apply image transformations
        if self.vision_processor:
            image = self.vision_processor(images=image, return_tensors='pt')['pixel_values'].reshape(3, 224, 224)
        
        return image, text, longitude, latitude

    def __len__(self):
        return len(self.text_data)


class im2gps3kDataset(VisionDataset):
    def __init__(self, root_path='./data/im2gps3k', text_data_path='im2gps3k_places365.csv', image_data_path='images/', vision_processor= None, text_processor=None):
        super().__init__(self)
        print('start loading im2gps...')
        self.root_path = root_path
        self.text_data_path = text_data_path
        self.image_data_path = image_data_path
        self.text_data = pd.read_csv(os.path.join(self.root_path, self.text_data_path))
        # self.text_data = self.text_data[self.text_data['IMG_ID'].str.endswith('.jpg')] # only keep jpg images
        print('read text data success')

        # location from str to float
        self.text_data.loc[:,'LAT'] = self.text_data['LAT'].astype(float)
        self.text_data.loc[:,'LON'] = self.text_data['LON'].astype(float)
        print('location from str to float success')

        self.vision_processor = vision_processor
        self.text_processor = text_processor

        self.tencrop = T.TenCrop(224)

    def __getitem__(self, index):
        image_path = self.text_data.iloc[index]['IMG_ID']
        text = image_path
        
        longitude = self.text_data.iloc[index]['LON']
        latitude = self.text_data.iloc[index]['LAT']

        image = Image.open(os.path.join(self.root_path, self.image_data_path, image_path))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # image = self.tencrop(image) # for tencrop
            
        if self.vision_processor:
            image = self.vision_processor(images=image, return_tensors='pt')['pixel_values'].reshape(-1,224,224)

        return image, text, longitude, latitude
    
    def __len__(self):
        return len(self.text_data)


class yfcc4kDataset(VisionDataset):
    def __init__(self, root_path='./data/yfcc4k', text_data_path='yfcc4k_places365.csv', image_data_path='images/', vision_processor= None, text_processor=None):
        super().__init__(self)
        print('start loading yfcc4k...')
        self.root_path = root_path
        self.text_data_path = text_data_path
        self.image_data_path = image_data_path
        self.text_data = pd.read_csv(os.path.join(self.root_path, self.text_data_path))
        # self.text_data = self.text_data[self.text_data['IMG_ID'].str.endswith('.jpg')] # only keep jpg images
        print('read text data success')

        # location from str to float
        self.text_data.loc[:,'LAT'] = self.text_data['LAT'].astype(float)
        self.text_data.loc[:,'LON'] = self.text_data['LON'].astype(float)
        print('location from str to float success')

        self.vision_processor = vision_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        image_path = self.text_data.iloc[index]['IMG_ID']
        text = image_path
        
        longitude = self.text_data.iloc[index]['LON']
        latitude = self.text_data.iloc[index]['LAT']

        image = Image.open(os.path.join(self.root_path, self.image_data_path, image_path))

        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        if self.vision_processor:
            image = self.vision_processor(images=image, return_tensors='pt')['pixel_values'].reshape(-1,224,224)

        return image, text, longitude, latitude
    
    def __len__(self):
        return len(self.text_data)
