import h5py

def print_h5_tree(obj, level=0):
    for key, val in obj.items():
        if isinstance(val, h5py.Group):
            print("  " * level + str(key) + " : Group")
            if level < 4:  # don't go too deep if not needed, but we need to find data
                print_h5_tree(val, level + 1)
        elif isinstance(val, h5py.Dataset):
            print("  " * level + str(key) + f" : Dataset, Shape: {val.shape}, Type: {val.dtype}")

try:
    with h5py.File("C:/tmp/test_pavlovian.nwb", "r") as f:
        print_h5_tree(f)
except Exception as e:
    print(f"File not ready or error: {e}")
