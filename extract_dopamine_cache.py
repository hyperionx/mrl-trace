import os
import glob
import requests
import numpy as np
import h5py
import json

DA_CACHE = os.environ.get("DA_CACHE", "/tmp/da_cache")
os.makedirs(DA_CACHE, exist_ok=True)
RAW_DIR = os.path.join(DA_CACHE, 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

def fetch_assets():
    print("Fetching Pavlovian assets from DANDI 000351...")
    res = requests.get('https://api.dandiarchive.org/api/dandisets/000351/versions/draft/assets/?page_size=100').json()
    pavlovian_assets = [r for r in res['results'] if 'Pavlovian' in r['path']]
    return pavlovian_assets

def download_asset(asset_id, out_file):
    if os.path.exists(out_file) and os.path.getsize(out_file) > 100 * 1024 * 1024:
        print(f"File {out_file} already exists, skipping download.")
        return
    dl_url = f"https://api.dandiarchive.org/api/dandisets/000351/versions/draft/assets/{asset_id}/download/"
    print(f"Downloading {asset_id} to {out_file}...")
    response = requests.get(dl_url, stream=True)
    response.raise_for_status()
    with open(out_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192*1024):
            if chunk:
                f.write(chunk)
    print(f"Downloaded {asset_id}.")

def extract_session(nwb_file, out_file):
    print(f"Extracting data from {nwb_file} to {out_file}...")
    with h5py.File(nwb_file, 'r') as f:
        # Get event descriptions and indices
        desc = f['acquisition/eventidx_table/event_description'][:]
        idx = f['acquisition/eventidx_table/event_index'][:]
        
        # Find index for 'Sound 1' and 'Fixed solenoid 3'
        sound_desc = b'Sound 1'
        rew_desc = b'Fixed solenoid 3'
        
        sound_idx = -1
        rew_idx = -1
        for i, d in enumerate(desc):
            if d == sound_desc:
                sound_idx = idx[i]
            elif d == rew_desc:
                rew_idx = idx[i]
                
        if sound_idx == -1 or rew_idx == -1:
            print("Could not find sound or reward events.")
            return False
            
        eventindex = f['acquisition/eventlog/eventindex'][:]
        eventtime = f['acquisition/eventlog/eventtime'][:]
        
        sound_t = eventtime[eventindex == sound_idx]
        reward_t = eventtime[eventindex == rew_idx]
        
        dff = f['processing/photometry/dff/data'][:]
        dff_t = f['processing/photometry/dff/timestamps'][:]
        
        # Approximate sampling rate
        fs = 1.0 / np.median(np.diff(dff_t))
        
        sub = f['general/subject/subject_id'][()]
        if isinstance(sub, bytes):
            sub = sub.decode('utf-8')
            
        sess_data = {
            'dff': dff,
            'dff_t': dff_t,
            'sound_t': sound_t,
            'reward_t': reward_t,
            'fs': fs,
            'sub': sub
        }
        
        np.save(out_file, sess_data)
        print(f"Extracted. Cues: {len(sound_t)}, Rewards: {len(reward_t)}")
        return True

def main():
    assets = fetch_assets()
    # Process just the first 2 assets to be quick but have enough trials
    for i in range(2):
        asset = assets[i]
        asset_id = asset['asset_id']
        raw_file = os.path.join(RAW_DIR, f"{asset_id}.nwb")
        cache_file = os.path.join(DA_CACHE, f"{asset_id}.npy")
        
        # We already downloaded one as test_pavlovian.nwb, we can reuse it if it matches
        if i == 0 and os.path.exists("C:/tmp/test_pavlovian.nwb"):
            print("Reusing existing downloaded file for the first asset.")
            # os.rename might fail across drives, so we just use the path
            raw_file = "C:/tmp/test_pavlovian.nwb"
            
        if not os.path.exists(cache_file):
            if raw_file != "C:/tmp/test_pavlovian.nwb":
                download_asset(asset_id, raw_file)
            extract_session(raw_file, cache_file)

if __name__ == '__main__':
    main()
