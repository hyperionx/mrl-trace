import urllib.request
url = "https://api.dandiarchive.org/api/dandisets/000351/versions/draft/assets/dab8b4f8-a54f-4eb3-98ec-fe143788e6b2/download/"
print("Downloading NWB file from", url)
urllib.request.urlretrieve(url, "test.nwb")
print("Downloaded test.nwb")
