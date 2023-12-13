import csv
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import requests
import pprint
import os
import time
import concurrent.futures
from tqdm import tqdm
from PIL import Image,ImageDraw,ImageFont

if __name__ == "__main__":

    tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER",
             "CHALLENGER"]
    ranks = [1, 2, 3, 4]

    summoner_names = []
    labels = []
    images = []

    for tier in tiers:
        for rank in ranks:
            if tier in ["CHALLENGER", "GRANDMASTER", "MASTER"] and rank != 1:
                continue
            print(tier, rank)

            warnings.filterwarnings('ignore')
            directory = f"{tier}/{rank}"
            try:
                file_list = os.listdir(directory)
            except:
                continue

            for file in file_list:
                file_path = directory + '/' + file
                data_df = pd.read_csv(file_path, encoding="ansi")
                summoner_names += data_df['summonerName'].tolist()
                labels += data_df['label'].tolist()

            for name in tqdm(summoner_names):
                text_width = 16 * 4
                text_height = 16 * 4
                canvas = Image.new('RGB', (text_width, text_height), "black")

                # 가운데에 그리기 (폰트 색: 하양)
                draw = ImageDraw.Draw(canvas)
                font_size = 16
                font = ImageFont.truetype("./NanumGothic-Regular.ttf", font_size)
                for i in range(len(name)):
                    for j in range(i, i + 4):
                        if len(name) < i:
                            continue
                        else:
                            draw.text((i % 4 * 16, i // 4 * 16 - 2), name[i], 'white', font)

                image = np.array(list(canvas.getdata()))
                image = image.reshape((text_height, text_width, 3))
                images.append(image)

    np.savez('preprocessed', names=summoner_names, labels=labels, images=images)