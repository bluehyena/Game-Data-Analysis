import requests
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import requests
import pprint
import os

pp = pprint.PrettyPrinter(indent=4)
api_key = 'RGAPI-a766c172-a012-4eea-8b71-3d20a7ef4e0a'
request_header = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.164 Safari/537.36",
                    "Accept-Language": "ko,en-US;q=0.9,en;q=0.8,es;q=0.7",
                    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
                    "Origin": "https://developer.riotgames.com",
                    "X-Riot-Token": api_key
}

def league_v4_queue_tier_division(queue, tier, division, page_number):
    if division == 1:
        division = 'I'
    elif division == 2 :
        division = 'II'
    elif division == 3 :
        division = 'III'
    elif division == 4:
        division = 'IV'
    if queue == "solo" :
        queue = "RANKED_SOLO_5x5"
    elif queue == "free" :
        queue = "RANKED_FLEX_SR"
    url = f"https://kr.api.riotgames.com/lol/league-exp/v4/entries/{queue}/{tier}/{division}?page={page_number}"
    return requests.get(url, headers=request_header)
    
def get_all_data(num, tier, rank):
    warnings.filterwarnings('ignore')
    directory = f"data/{tier}/{rank}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    req = league_v4_queue_tier_division("solo", tier, rank, num)

    if req.status_code == 200:
        all_data = req.json()
        print(all_data)
        if len(all_data) == 0:
            return -2

        data_df = pd.DataFrame(all_data)

        data_df.to_csv(f"data/{tier}/{rank}/{tier}{rank}_{i}.csv", mode='w', encoding="ansi", index=False)
        return -1
    else:
        return num

if __name__ == "__main__":
    
    tiers = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]
    ranks = [1, 2, 3, 4]

    for tier in tiers:
        for rank in ranks:
            if tier in ["CHALLENGER", "GRANDMASTER", "MASTER"] and rank != 1:
                continue
            i = 1
            while True:
                if get_all_data(i, tier, rank) == -1:
                    i += 1
                if get_all_data(i, tier, rank) == -2:
                    break