# changes loading model from checkpoint, unwrapping
from importlib import metadata
import torch
import os
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import MP16Dataset
from utils.G3 import G3
from accelerate import Accelerator, DistributedDataParallelKwargs
import warnings
import webdataset as wds
import pandas as pd

warnings.filterwarnings('ignore')

def train_1epoch(dataloader, eval_dataloader, earlystopper, model, vision_processor, text_processor, optimizer, scheduler, device, accelerator=None):
    model.train()
    t = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    for i, (images, texts, longitude, latitude) in enumerate(t):

        texts = text_processor(text=texts, padding='max_length', truncation=True, return_tensors='pt', max_length=77)
        images = images.to(device)
        texts = texts.to(device)
        longitude = longitude.to(device).float()
        latitude = latitude.to(device).float()
        optimizer.zero_grad()

        output = model(images, texts, longitude, latitude, return_loss=True)
        loss = output['loss']

        # loss.backward()
        accelerator.backward(loss)
        optimizer.step()
        if i % 1 == 0:
            t.set_description('step {}, loss {}, lr {}'.format(i, loss.item(), scheduler.get_last_lr()[0]))
    scheduler.step()

def filter_function(sample, metadata_dict):
    key = sample["__key__"]  # The first item in sample is the key
    filename = key.split("/")[-1]  # Extract filename from key
    return filename in metadata_dict

def preprocess(sample, model):
    print("Preprocessing")
    img, text, lon, lat = sample
    # Apply vision processor
    img = model.vision_processor(images=img, return_tensors='pt')['pixel_values'].squeeze(0)
    # Apply text processor
    text_inputs = model.text_processor(text=[text], padding='max_length', truncation=True, return_tensors='pt', max_length=77)
    return img, text_inputs, lon, lat

def add_mp_metadata(sample, metadata_dict):
    key = sample['__key__']
    print(key)
    filename = key.split('/')[-1]  # Extract the filename
    if filename in metadata_dict:
        text, lon, lat = metadata_dict[filename]
        sample['text'] = text
        sample['longitude'] = lon
        sample['latitude'] = lat
    return sample

def create_mp_metadata():
    text_data = pd.read_csv('./data/remaining_dataset.csv')
    text_data['IMG_ID'] = text_data['IMG_ID'].apply(lambda x: x.replace('/', '_'))

    # Create a dictionary mapping filename to (text, longitude, latitude)
    metadata_dict = {}
    for i, row in text_data.iterrows():
        img_id = row['IMG_ID']  # After normalization: '/' replaced with '_'
        print(img_id)
        longitude = float(row['LON'])
        latitude = float(row['LAT'])
        # Build location text description
        location_elements = [row[c] for c in ['neighbourhood','city','state','country'] if pd.notna(row[c])]
        text = 'A street view photo taken in ' + ', '.join(location_elements)
        metadata_dict[img_id] = (text, longitude, latitude)
        return metadata_dict

def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # fine-tune
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    file_path = './checkpoints/g3.pth'
    model = G3(device).to(device)
    if os.path.exists(file_path):
        state_dict = torch.load(file_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model state loaded successfully.")
    else:
        print(f"File does not exist. Skipping model load.")
    location_encoder_dict = torch.load('location_encoder.pth') # from geoclip
    model.location_encoder.load_state_dict(location_encoder_dict)

    metadata_dict = create_mp_metadata()

    wds_dataset = (
        wds.WebDataset("./data/mp-16-images.tar")
        .select(filter_function)
        .decode("pil")
        .to_tuple("jpg", "text", "longitude", "latitude")
    )
    
    wds_dataset = wds_dataset.map(add_mp_metadata, metadata_dict=metadata_dict)
    wds_dataset = wds_dataset.map(preprocess, model=model)
    
    # dataset = MP16Dataset(wds_dataset, vision_processor = model.vision_processor, text_processor = model.text_processor)
    dataloader = wds.WebLoader(wds_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=5)

    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
            params.append(param)

    optimizer = torch.optim.AdamW([param for name,param in model.named_parameters() if param.requires_grad], lr=3e-5, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.87)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    unwrapped_model = accelerator.unwrap_model(model)
    vision_processor = unwrapped_model.vision_processor
    text_processor = unwrapped_model.text_processor

    eval_dataloader = None
    earlystopper = None
    for epoch in range(10):
        train_1epoch(dataloader, eval_dataloader, earlystopper, model, vision_processor, text_processor, optimizer, scheduler, device, accelerator)
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), './checkpoints/g3.pth')

if __name__ == '__main__':
    main()
