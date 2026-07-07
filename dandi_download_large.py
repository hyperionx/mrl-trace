import requests
import json
import os

os.makedirs('C:/tmp', exist_ok=True)
out_file = 'C:/tmp/test_pavlovian.nwb'

print("Getting asset list...")
res = requests.get('https://api.dandiarchive.org/api/dandisets/000351/versions/draft/assets/?page_size=100').json()
pavlovian_assets = [r for r in res['results'] if 'Pavlovian' in r['path']]
asset_id = pavlovian_assets[0]['asset_id']

print("Getting download URL...")
dl_url = f"https://api.dandiarchive.org/api/dandisets/000351/versions/draft/assets/{asset_id}/download/"

print(f"Downloading {dl_url} to {out_file}...")
response = requests.get(dl_url, stream=True)
response.raise_for_status()

downloaded = 0
with open(out_file, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192*1024):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)
            print(f"Downloaded {downloaded / 1024 / 1024:.2f} MB")

print("Download complete.")
