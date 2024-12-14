import faiss
import torch
import numpy as np
import os
import argparse
import pandas as pd
from PIL import Image
from geopy.distance import geodesic
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from utils.G3 import G3

class GeoImageDataset(Dataset):
    def __init__(self, dataframe, img_folder, topn, vision_processor, database_df, I):
        self.dataframe = dataframe
        self.img_folder = img_folder
        self.topn = topn
        self.vision_processor = vision_processor
        self.database_df = database_df
        self.I = I

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = f'{self.img_folder}/{self.dataframe.loc[idx, "IMG_ID"]}'
        image = Image.open(img_path).convert('RGB')
        image = self.vision_processor(images=image, return_tensors='pt')['pixel_values'].reshape(3,224,224)
        
        gps_data = []
        search_top1_latitude, search_top1_longitude = self.database_df.loc[self.I[idx][0], ['LAT', 'LON']].values
        rag_5, rag_10, rag_15, zs = [],[],[],[]
        for j in range(self.topn):
            gps_data.extend([
                float(self.dataframe.loc[idx, f'5_rag_{j}_latitude']),
                float(self.dataframe.loc[idx, f'5_rag_{j}_longitude']),
                float(self.dataframe.loc[idx, f'10_rag_{j}_latitude']),
                float(self.dataframe.loc[idx, f'10_rag_{j}_longitude']),
                float(self.dataframe.loc[idx, f'15_rag_{j}_latitude']),
                float(self.dataframe.loc[idx, f'15_rag_{j}_longitude']),
                float(self.dataframe.loc[idx, f'zs_{j}_latitude']),
                float(self.dataframe.loc[idx, f'zs_{j}_longitude']),
                search_top1_latitude,
                search_top1_longitude
            ])
        
        gps_data = np.array(gps_data).reshape(-1, 2)
        return image, gps_data, idx

def evaluate(args, I):
    print('start evaluation')
    if args.database == 'mp16':
        database = args.database_df
        df = args.dataset_df
        df['NN_idx'] = I[:, 0]
        df['LAT_pred'] = df.apply(lambda x: database.loc[x['NN_idx'],'LAT'], axis=1)
        df['LON_pred'] = df.apply(lambda x: database.loc[x['NN_idx'],'LON'], axis=1)

        df_llm = pd.read_csv(f'./data/{args.dataset}/{args.dataset}_prediction.csv')
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = G3(device).to(device)
        state_dict = torch.load('./checkpoints/g3.pth', map_location=device)
        model.load_state_dict(state_dict)
        topn = 5 # number of candidates

        dataset = GeoImageDataset(df_llm, f'./data/{args.dataset}/images', topn, vision_processor=model.vision_processor, database_df=database, I=I)
        data_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=6, pin_memory=True)

        for images, gps_batch, indices in tqdm(data_loader):
            images = images.to(args.device)
            image_embeds = model.vision_projection_else_2(model.vision_projection(model.vision_model(images)[1]))
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True) # b, 768

            gps_batch = gps_batch.to(args.device)
            gps_input = gps_batch.clone().detach()
            b, c, _ = gps_input.shape
            gps_input = gps_input.reshape(b*c, 2)
            location_embeds = model.location_encoder(gps_input)
            location_embeds = model.location_projection_else(location_embeds.reshape(b*c, -1))
            location_embeds = location_embeds / location_embeds.norm(p=2, dim=-1, keepdim=True)
            location_embeds = location_embeds.reshape(b, c, -1) #  b, c, 768

            similarity = torch.matmul(image_embeds.unsqueeze(1), location_embeds.permute(0, 2, 1)) # b, 1, c
            similarity = similarity.squeeze(1).cpu().detach().numpy()
            max_idxs = np.argmax(similarity, axis=1)
            
            # update DataFrame
            for i, max_idx in enumerate(max_idxs):
                final_idx = indices[i]
                final_idx = final_idx.item()
                final_latitude, final_longitude = gps_batch[i][max_idx]
                final_latitude, final_longitude = final_latitude.item(), final_longitude.item()
                if final_latitude < -90 or final_latitude > 90:
                    final_latitude = 0
                if final_longitude < -180 or final_longitude > 180:
                    final_longitude = 0
                df.loc[final_idx, 'LAT_pred'] = final_latitude
                df.loc[final_idx, 'LON_pred'] = final_longitude

        df_cleaned = df.dropna(subset=['LAT', 'LON', 'LAT_pred', 'LON_pred'])
        df_cleaned['geodesic'] = df_cleaned.apply(lambda x: geodesic((x['LAT'], x['LON']), (x['LAT_pred'], x['LON_pred'])).km, axis=1)
        print(df_cleaned.head())
        df_cleaned.to_csv(f'./data/{args.dataset}_{args.index}_results.csv', index=False)
        df['geodesic'] = df_cleaned['geodesic']
        # 1, 25, 200, 750, 2500 km level
        print('2500km level: ', df[df['geodesic'] < 2500].shape[0] / df.shape[0])
        print('750km level: ', df[df['geodesic'] < 750].shape[0] / df.shape[0])
        print('200km level: ', df[df['geodesic'] < 200].shape[0] / df.shape[0])
        print('25km level: ', df[df['geodesic'] < 25].shape[0] / df.shape[0])
        print('1km level: ', df[df['geodesic'] < 1].shape[0] / df.shape[0])

if __name__ == '__main__':

    res = faiss.StandardGpuResources()

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, default='g3')
    parser.add_argument('--dataset', type=str, default='im2gps3k')
    parser.add_argument('--database', type=str, default='mp16')
    args = parser.parse_args()
    if args.dataset == 'im2gps3k':
        args.dataset_df = pd.read_csv('./data/im2gps3k/im2gps3k_places365.csv')
    elif args.dataset == 'yfcc4k':
        args.dataset_df = pd.read_csv('./data/yfcc4k/yfcc4k_places365.csv')

    if args.database == 'mp16':
        args.database_df = pd.read_csv('./data/MP16_Pro_filtered.csv')

    args.device = "cuda" if torch.cuda.is_available() else "cpu"


    D = np.load(f'./index/D_{args.index}_{args.dataset}.npy')
    I = np.load(f'./index/I_{args.index}_{args.dataset}.npy')
    evaluate(args, I)

